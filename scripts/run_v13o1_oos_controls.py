#!/usr/bin/env python3
"""
V13O.1: out-of-sample zeta-window validation and constrained-potential controls for the V13
finite-dimensional renormalized DTES–Hilbert–Pólya style candidate family.

  python3 scripts/run_v13o1_oos_controls.py \\
    --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\
    --formula_json runs/v13_operator_formula/formula_components_summary.json \\
    --v13n_summary runs/v13n_theorem_report/v13n_summary.json \\
    --out_dir runs/v13o1_oos_controls \\
    --seed 42 --max_iter 300 --tol 1e-3

This is computational evidence only; not a proof of RH. Language: candidate approximant,
out-of-sample evidence, specificity risk — not a Hilbert–Pólya operator in the classical sense.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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

TARGET_GROUPS = [
    "real_zeta",
    "shuffled_zeta",
    "reversed_zeta",
    "GUE_synthetic_targets",
    "Poisson_targets",
]

DIM_CONFIGS: List[Dict[str, Any]] = [
    {
        "dim": 64,
        "k_train": 45,
        "lambda_p": 2.121320343559643,
        "geo_sigma": 0.5045378491522287,
        "smooth_sigma": 0.7071067811865476,
        "beta0": 0.3,
        "tau": 200.0,
        "beta_floor": 0.03,
    },
    {
        "dim": 128,
        "k_train": 96,
        "lambda_p": 3.0,
        "geo_sigma": 0.6,
        "smooth_sigma": 1.0,
        "beta0": 0.3,
        "tau": 500.0,
        "beta_floor": 0.03,
    },
    {
        "dim": 256,
        "k_train": 128,
        "lambda_p": 2.121320343559643,
        "geo_sigma": 0.5045378491522287,
        "smooth_sigma": 1.25,
        "beta0": 0.3,
        "tau": 300.0,
        "beta_floor": 0.03,
    },
]

FREEZE_MODES = ("full_H", "fixed_V", "fixed_K_update_V_test")


def potential_variant_defs() -> List[Dict[str, Any]]:
    return [
        {
            "id": "full_V",
            "lambda_scale": 1.0,
            "smooth_scale": 1.0,
            "clip": CLIP_DEFAULT,
            "freeze_v_after": None,
            "variant": "",
            "target_blind": False,
        },
        {
            "id": "weak_V",
            "lambda_scale": 0.5,
            "smooth_scale": 1.0,
            "clip": CLIP_DEFAULT,
            "freeze_v_after": None,
            "variant": "",
            "target_blind": False,
        },
        {
            "id": "very_weak_V",
            "lambda_scale": 0.25,
            "smooth_scale": 1.0,
            "clip": CLIP_DEFAULT,
            "freeze_v_after": None,
            "variant": "",
            "target_blind": False,
        },
        {
            "id": "frozen_V_after_5",
            "lambda_scale": 1.0,
            "smooth_scale": 1.0,
            "clip": CLIP_DEFAULT,
            "freeze_v_after": 5,
            "variant": "",
            "target_blind": False,
        },
        {
            "id": "frozen_V_after_10",
            "lambda_scale": 1.0,
            "smooth_scale": 1.0,
            "clip": CLIP_DEFAULT,
            "freeze_v_after": 10,
            "variant": "",
            "target_blind": False,
        },
        {
            "id": "smooth_V_strong",
            "lambda_scale": 1.0,
            "smooth_scale": 2.0,
            "clip": CLIP_DEFAULT,
            "freeze_v_after": None,
            "variant": "",
            "target_blind": False,
        },
        {
            "id": "clipped_V_strong",
            "lambda_scale": 1.0,
            "smooth_scale": 1.0,
            "clip": (5.0, 95.0),
            "freeze_v_after": None,
            "variant": "",
            "target_blind": False,
        },
        {
            "id": "lowfreq_V",
            "lambda_scale": 1.0,
            "smooth_scale": 1.0,
            "clip": CLIP_DEFAULT,
            "freeze_v_after": None,
            "variant": "lowfreq_V",
            "target_blind": False,
        },
        {
            "id": "target_blind_V",
            "lambda_scale": 1.0,
            "smooth_scale": 1.0,
            "clip": CLIP_DEFAULT,
            "freeze_v_after": None,
            "variant": "",
            "target_blind": True,
            "experimental": True,
        },
    ]


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
    t = t.replace("#", "\\#")
    t = t.replace("&", "\\&")
    t = t.replace("$", "\\$")
    t = t.replace("^", "\\textasciicircum{}")
    t = t.replace("~", "\\textasciitilde{}")
    return t


def gue_positive_ordinates(k: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((k, k))
    h = (a + a.T) / math.sqrt(2.0)
    w = np.sort(np.linalg.eigvalsh(h).astype(np.float64))
    w = w - float(w[0]) + 1e-6
    return w.astype(np.float64)


def poisson_positive_ordinates(k: int, rng: np.random.Generator) -> np.ndarray:
    s = rng.exponential(1.0, size=(k,)).astype(np.float64)
    c = np.cumsum(s)
    c = c - float(c[0]) + 1e-6
    return c


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


def train_test_windows(
    target_group: str,
    z_sorted_global: np.ndarray,
    k_train: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """Return (train_ordinates, test_ordinates, train_ordered, notes)."""
    need = 2 * int(k_train)
    base = np.asarray(z_sorted_global[:need], dtype=_DTF).reshape(-1)
    if target_group == "real_zeta":
        return base[:k_train].copy(), base[k_train:need].copy(), False, "sorted_zeta_pool_slice"
    if target_group == "shuffled_zeta":
        u = base.copy()
        rng.shuffle(u)
        return u[:k_train].copy(), u[k_train:need].copy(), True, "single_shuffled_split"
    if target_group == "reversed_zeta":
        rev = base[::-1].copy()
        return rev[:k_train].copy(), rev[k_train:need].copy(), True, "reversed_pool_prefix"
    if target_group == "GUE_synthetic_targets":
        g = gue_positive_ordinates(need, rng)
        return g[:k_train].copy(), g[k_train:need].copy(), False, "GUE_iid_matrix"
    if target_group == "Poisson_targets":
        p = poisson_positive_ordinates(need, rng)
        return p[:k_train].copy(), p[k_train:need].copy(), False, "Poisson_unfolding"
    raise ValueError(target_group)


def meets_train_acceptance(
    *,
    od: float,
    sp: float,
    ks: float,
    finite: bool,
    sa: float,
    eig_error: Optional[str],
) -> bool:
    if eig_error is not None:
        return False
    if not math.isfinite(od) or od > 1e-3:
        return False
    if not math.isfinite(sp) or sp > 1.2:
        return False
    if not math.isfinite(ks) or ks > 0.25:
        return False
    if not finite:
        return False
    if not math.isfinite(sa) or sa > 1e-12:
        return False
    return True


def meets_test_acceptance(*, sp: float, ks: float, finite: bool, sa: float, eig_error: Optional[str]) -> bool:
    if eig_error is not None:
        return False
    if not math.isfinite(sp) or sp > 1.5:
        return False
    if not math.isfinite(ks) or ks > 0.30:
        return False
    if not finite:
        return False
    if not math.isfinite(sa) or sa > 1e-12:
        return False
    return True


def window_rng(
    seed: int, dim: int, operator_key: str, target_group: str, *, rep_idx: int = 0
) -> np.random.Generator:
    h = int.from_bytes(
        hashlib.sha256(f"{dim}|{operator_key}|{target_group}|{rep_idx}".encode("utf-8")).digest()[:4],
        "little",
    )
    return np.random.default_rng((int(seed) ^ int(h)) % (2**32 - 1))


def fmt_time(sec: float) -> str:
    if sec is None or not math.isfinite(float(sec)):
        return "?"
    sec_f = max(float(sec), 0.0)
    if sec_f < 1.0:
        return f"{sec_f:.2f}s"
    s_int = int(sec_f)
    h, rem = divmod(s_int, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def n_repeats_for(target_group: str, operator_group: str, args: Any) -> int:
    if target_group in {"shuffled_zeta", "GUE_synthetic_targets", "Poisson_targets"}:
        return max(1, int(args.n_controls))
    if operator_group in {"random_words", "random_words_n30", "random_words_n5"}:
        # Distinct random words are already enumerated via --n_random_words slots; avoid N^2 jobs.
        if operator_group in {"random_words_n30", "random_words_n5"}:
            return 1
        return max(1, int(args.n_random_words))
    return 1


def operator_groups_flat(n_rw: int, n_sym: int) -> List[str]:
    g = [
        "primary_word_seed6",
        "rejected_word_seed17",
        "ablate_K",
        "ablate_V",
        "ablate_L",
    ]
    g += ["random_words_n30"] * int(max(0, n_rw))
    g += ["random_symmetric_baseline"] * int(max(0, n_sym))
    return g


def compute_total_jobs(
    dim_cfgs_: List[Dict[str, Any]],
    pot_defs_: List[Dict[str, Any]],
    targets_: List[str],
    n_rw_: int,
    n_sym_: int,
    args_: Any,
) -> int:
    tot = 0
    ogs = operator_groups_flat(n_rw_, n_sym_)
    for _cfg in dim_cfgs_:
        for _pdef in pot_defs_:
            for tg in targets_:
                for og in ogs:
                    tot += n_repeats_for(tg, og, args_)
    return tot


def should_log_job(job_idx: int, total_jobs: int, every: int) -> bool:
    if total_jobs <= 0:
        return True
    if job_idx == 1 or job_idx == total_jobs:
        return True
    return every > 0 and (job_idx % every == 0)


def classify_v13o1(runs: List[Dict[str, Any]], group_summary: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    """Return (classification, main_conclusion, next_step)."""

    def filt(**kw: Any) -> List[Dict[str, Any]]:
        out = runs
        for k, v in kw.items():
            out = [r for r in out if r.get(k) == v]
        return out

    primary_full = filt(
        target_group="real_zeta",
        operator_group="primary_word_seed6",
        potential_variant="full_V",
        freeze_mode="full_H",
    )
    real_pass_by_dim: Dict[int, bool] = {}
    for d in (64, 128, 256):
        rs = [r for r in primary_full if int(r["dim"]) == d]
        real_pass_by_dim[d] = bool(rs) and all(bool(r.get("accepted_train")) and bool(r.get("accepted_test")) for r in rs)

    n_ok = sum(1 for d in (64, 128, 256) if real_pass_by_dim.get(d))

    def grp_rate(tg: str, og: Optional[str] = None) -> float:
        rows = [g for g in group_summary if g.get("target_group") == tg]
        if og:
            rows = [g for g in rows if g.get("operator_group") == og]
        if not rows:
            return float("nan")
        return float(np.mean([float(g.get("accepted_test_rate", float("nan"))) for g in rows]))

    rz = grp_rate("real_zeta", "primary_word_seed6")
    gue = grp_rate("GUE_synthetic_targets")
    poi = grp_rate("Poisson_targets")
    shu = grp_rate("shuffled_zeta")
    abl_l = grp_rate("real_zeta", "ablate_L")
    abl_v = grp_rate("real_zeta", "ablate_V")
    full_r = grp_rate("real_zeta", "primary_word_seed6")

    gaps = [
        float(r.get("generalization_gap_J", float("nan")))
        for r in primary_full
        if math.isfinite(float(r.get("generalization_gap_J", float("nan"))))
    ]
    med_gap = float(np.median(np.asarray(gaps, dtype=np.float64))) if gaps else float("nan")

    def min_mean_J(tg: str, og: str) -> float:
        rows = [g for g in group_summary if g.get("target_group") == tg and g.get("operator_group") == og]
        if not rows:
            return float("nan")
        return float(np.min([float(g.get("mean_J_test", float("nan"))) for g in rows]))

    J_real = min_mean_J("real_zeta", "primary_word_seed6")
    J_abL = min_mean_J("real_zeta", "ablate_L")
    outperform_abL = math.isfinite(J_real) and math.isfinite(J_abL) and J_abL < J_real * 0.98

    next_default = "V13P analytic renormalization or V13O.2 target-blind controls"

    if n_ok == 0:
        return (
            "failed_specificity",
            "The primary real-zeta configuration did not meet joint train and test acceptance on any tested dimension; out-of-sample evidence does not support specificity.",
            next_default,
        )

    controls_lower = (
        math.isfinite(rz)
        and math.isfinite(gue)
        and math.isfinite(poi)
        and math.isfinite(shu)
        and gue < rz - 1e-6
        and poi < rz - 1e-6
        and shu < rz - 0.02
    )
    ablations_ok = not outperform_abL and (not math.isfinite(abl_v) or abl_v <= full_r + 0.05)
    gap_ok = (not math.isfinite(med_gap)) or (abs(med_gap) < 3.0)

    if n_ok == 3 and controls_lower and ablations_ok and gap_ok:
        return (
            "strong_specificity",
            "Across dimensions, the primary candidate shows train and test acceptance while several synthetic targets and control operators exhibit lower out-of-sample acceptance rates; ablations do not clearly outperform the full mixing law under the reported metrics.",
            next_default,
        )

    if n_ok >= 2:
        return (
            "medium_specificity",
            "Real-zeta test acceptance holds on at least two dimensions, but GUE/Poisson or ablation controls remain partially competitive; interpret as moderate specificity risk.",
            next_default,
        )

    if n_ok >= 1:
        return (
            "weak_specificity",
            "Train metrics may pass while test acceptance is inconsistent across dimensions, or synthetic controls match real-zeta performance on out-of-sample windows.",
            next_default,
        )

    return (
        "failed_specificity",
        "Out-of-sample generalization for the primary real-zeta line is weak or absent under current thresholds.",
        next_default,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.1 OOS zeta-window validation and constrained potentials.")
    ap.add_argument("--candidate_json", type=str, default="runs/v13_candidate_theorem/candidate_operators.json")
    ap.add_argument("--formula_json", type=str, default="runs/v13_operator_formula/formula_components_summary.json")
    ap.add_argument("--v13n_summary", type=str, default="runs/v13n_theorem_report/v13n_summary.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13o1_oos_controls")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iter", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--n_controls", type=int, default=30, help="Repeats for shuffled/GUE/Poisson target draws.")
    ap.add_argument("--n_random_words", type=int, default=30, help="Number of distinct random-word operator slots.")
    ap.add_argument("--n_random_sym", type=int, default=5)
    ap.add_argument("--progress_every", type=int, default=1, help="Log every N jobs (always logs first and last).")
    ap.add_argument(
        "--verbose_iter_every",
        type=int,
        default=0,
        help="Print inner fixed-point progress every N iterations (0 = off).",
    )
    ap.add_argument("--smoke", action="store_true", help="Tiny grid for CI / smoke (subset of dims, targets, operators).")
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_examples = """\
Smoke test:
python3 scripts/run_v13o1_oos_controls.py \\
  --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\
  --formula_json runs/v13_operator_formula/formula_components_summary.json \\
  --v13n_summary runs/v13n_theorem_report/v13n_summary.json \\
  --out_dir runs/v13o1_oos_controls_smoke \\
  --seed 42 \\
  --max_iter 60 \\
  --tol 1e-3 \\
  --n_controls 5 \\
  --n_random_words 5 \\
  --progress_every 1

Full run:
caffeinate -dimsu python3 scripts/run_v13o1_oos_controls.py \\
  --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\
  --formula_json runs/v13_operator_formula/formula_components_summary.json \\
  --v13n_summary runs/v13n_theorem_report/v13n_summary.json \\
  --out_dir runs/v13o1_oos_controls \\
  --seed 42 \\
  --max_iter 300 \\
  --tol 1e-3 \\
  --n_controls 30 \\
  --n_random_words 30 \\
  --progress_every 1
"""
    print(run_examples, flush=True)

    import importlib.util

    def _load_v13_validate() -> Any:
        path = Path(ROOT) / "scripts" / "validate_candidates_v13.py"
        spec = importlib.util.spec_from_file_location("_v13_validate", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_v13_validate_oos"] = mod
        spec.loader.exec_module(mod)
        return mod

    v = _load_v13_validate()
    from core import v13l_self_consistent as v13l
    from core import v13l1_stabilized as v13l1
    from core import v13o1_oos as oos
    from core.artin_operator import sample_domain
    from core.v13m1_renormalized import spectral_metrics_window as smw

    cand = _load_json(_resolve(args.candidate_json))
    form = _load_json(_resolve(args.formula_json))
    v13n = _load_json(_resolve(args.v13n_summary))

    max_k = max(int(c["k_train"]) for c in DIM_CONFIGS)
    n_zeros = max(512, 2 * max_k)
    z_pool = v._load_zeros(int(n_zeros))
    z_sorted = np.asarray(z_pool, dtype=_DTF).reshape(-1)
    z_sorted = z_sorted[np.isfinite(z_sorted) & (z_sorted > 0.0)]
    z_sorted = np.sort(z_sorted)

    base_seed = int(args.seed)
    all_runs: List[Dict[str, Any]] = []

    dim_cfgs = list(DIM_CONFIGS)
    pot_defs = potential_variant_defs()
    targets = list(TARGET_GROUPS)

    if args.smoke:
        dim_cfgs = [c for c in DIM_CONFIGS if int(c["dim"]) == 64]
        pot_defs = [p for p in pot_defs if p["id"] in ("full_V", "weak_V", "target_blind_V")]
        targets = ["real_zeta", "GUE_synthetic_targets"]
        freeze_modes = ("full_H",)
    else:
        freeze_modes = FREEZE_MODES

    n_rw = max(1, int(args.n_random_words))
    n_sym = max(1, int(args.n_random_sym))

    def random_word(rng: np.random.Generator) -> List[int]:
        alphabet = list(range(-6, 0)) + list(range(1, 7))
        return [int(rng.choice(alphabet)) for _ in range(len(PRIMARY_WORD))]

    total_jobs = compute_total_jobs(dim_cfgs, pot_defs, targets, n_rw, n_sym, args)
    job_idx = 0
    t0 = time.perf_counter()
    prog_every = max(1, int(args.progress_every))
    ve = int(args.verbose_iter_every)

    def make_on_train_iter() -> Optional[Callable[[int, int, Dict[str, Any]], None]]:
        if ve <= 0:
            return None
        mi = int(args.max_iter)

        def on_train_iter(it: int, max_iter_: int, info: Dict[str, Any]) -> None:
            if it == 0 or (it + 1) % ve == 0 or (it + 1) == max_iter_ or (it + 1) == mi:
                od = float(info.get("operator_diff", float("nan")))
                sl = float(info.get("spectral_log_mse", float("nan")))
                sp = float(info.get("spacing_mse_normalized", float("nan")))
                ks = float(info.get("ks_wigner", float("nan")))
                jtr = float(info.get("pareto_objective", float("nan")))
                print(
                    f"    [iter] {it+1}/{max_iter_} op_diff={od:.3e} "
                    f"J_train={jtr:.6g} spectral_log_mse={sl:.6g} spacing_mse_normalized={sp:.6g} ks_wigner={ks:.6g}",
                    flush=True,
                )

        return on_train_iter

    on_train_cb = make_on_train_iter()

    def append_failed_rows(
        *,
        err_msg: str,
        dim_: int,
        k_train_: int,
        target_group_: str,
        op: Dict[str, Any],
        op_sub_: str,
        pdef: Dict[str, Any],
        word_json_: str,
        win_note_: str,
        train_ordered_: bool,
        lp_: float,
        sm_: float,
        clip_lo_: float,
        clip_hi_: float,
    ) -> None:
        nanf = float("nan")
        for fmode in freeze_modes:
            all_runs.append(
                {
                    "dim": dim_,
                    "k_train": k_train_,
                    "target_group": target_group_,
                    "operator_group": op["group"],
                    "operator_sub_id": op_sub_,
                    "potential_variant": pdef["id"],
                    "potential_experimental": bool(pdef.get("experimental", False)),
                    "freeze_mode": fmode,
                    "window_note": win_note_,
                    "train_ordered": train_ordered_,
                    "lambda_p_effective": lp_,
                    "smooth_sigma_effective": sm_,
                    "clip_percentiles": json.dumps([clip_lo_, clip_hi_]),
                    "word": word_json_,
                    "spectral_log_mse_train": nanf,
                    "spacing_mse_normalized_train": nanf,
                    "ks_wigner_train": nanf,
                    "pareto_objective_train": nanf,
                    "spectral_log_mse_test": nanf,
                    "spacing_mse_normalized_test": nanf,
                    "ks_wigner_test": nanf,
                    "pareto_objective_test": nanf,
                    "generalization_gap_J": nanf,
                    "generalization_ratio_J": nanf,
                    "operator_diff_final": nanf,
                    "operator_diff_test_mode": nanf,
                    "self_adjointness_fro": nanf,
                    "finite": False,
                    "accepted_train": False,
                    "accepted_test": False,
                    "eig_error": err_msg,
                    "n_iter": 0,
                    "converged_operator": False,
                }
            )

    for cfg in dim_cfgs:
        dim = int(cfg["dim"])
        k_train = int(cfg["k_train"])
        if int(z_sorted.size) < 2 * k_train:
            raise RuntimeError(f"Need at least {2*k_train} zeta ordinates; got {z_sorted.size}.")

        Z = sample_domain(dim, seed=base_seed)

        operator_jobs: List[Dict[str, Any]] = [
            {"group": "primary_word_seed6", "word": list(PRIMARY_WORD), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "word"},
            {"group": "rejected_word_seed17", "word": list(REJECTED_WORD), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "word"},
            {"group": "ablate_K", "word": list(PRIMARY_WORD), "lap": 1.0, "geo_w": 0.0, "kind": "word"},
            {"group": "ablate_V", "word": list(PRIMARY_WORD), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "ablate_V"},
            {"group": "ablate_L", "word": list(PRIMARY_WORD), "lap": 0.0, "geo_w": GEO_WEIGHT, "kind": "word"},
        ]
        for j in range(n_rw):
            operator_jobs.append(
                {
                    "group": "random_words_n30",
                    "word": random_word(np.random.default_rng(base_seed + dim * 7919 + j)),
                    "lap": 1.0,
                    "geo_w": GEO_WEIGHT,
                    "kind": "word",
                    "sub_id": f"rw_{j}",
                }
            )
        fro_t_ref = float(
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
        for j in range(n_sym):
            r = np.random.default_rng(base_seed + dim * 104729 + j).standard_normal((dim, dim))
            S = 0.5 * (r + r.T)
            fs = float(np.linalg.norm(S, ord="fro"))
            S = (S * (fro_t_ref / max(fs, 1e-12))).astype(_DTF, copy=False)
            operator_jobs.append({"group": "random_symmetric_baseline", "matrix": S, "kind": "sym", "sub_id": f"sym_{j}"})

        for pdef in pot_defs:
            lp = float(cfg["lambda_p"]) * float(pdef["lambda_scale"])
            sm = float(cfg["smooth_sigma"]) * float(pdef["smooth_scale"])
            clip_lo, clip_hi = float(pdef["clip"][0]), float(pdef["clip"][1])
            variant = str(pdef.get("variant") or "")
            freeze_v_after = pdef.get("freeze_v_after")
            target_blind = bool(pdef.get("target_blind", False))

            for target_group in targets:
                for op in operator_jobs:
                    n_rep = n_repeats_for(target_group, op["group"], args)
                    if op["kind"] == "sym":
                        H_base = np.asarray(op["matrix"], dtype=_DTF, copy=True)
                        word_json = json.dumps([])
                    else:
                        H_base = build_h(
                            z_points=Z,
                            word=list(op["word"]),
                            v13l=v13l,
                            geo_sigma=float(cfg["geo_sigma"]),
                            laplacian_weight=float(op["lap"]),
                            geo_weight=float(op["geo_w"]),
                        )
                        word_json = json.dumps([int(x) for x in op["word"]])

                    op_sub = op.get("sub_id", op["group"])
                    operator_key = f"{op['group']}:{op_sub}"

                    for rep in range(n_rep):
                        job_idx += 1
                        elapsed = time.perf_counter() - t0
                        avg = elapsed / max(job_idx, 1)
                        remaining = avg * max(total_jobs - job_idx, 0)
                        if should_log_job(job_idx, total_jobs, prog_every):
                            print(
                                f"[v13o1] ({job_idx}/{total_jobs}) "
                                f"dim={dim} target={target_group} op={op['group']} "
                                f"V={pdef['id']} rep={rep} "
                                f"elapsed={fmt_time(elapsed)} eta={fmt_time(remaining)}",
                                flush=True,
                            )

                        try:
                            tr_ord, te_ord, train_ordered, win_note = train_test_windows(
                                target_group,
                                z_sorted,
                                k_train,
                                window_rng(base_seed, dim, operator_key, target_group, rep_idx=rep),
                            )
                            zeros_train_metric = np.sort(np.asarray(tr_ord, dtype=_DTF).reshape(-1))
                            zeros_test_metric = np.sort(np.asarray(te_ord, dtype=_DTF).reshape(-1))
                            k_align_tr = int(min(dim, k_train, zeros_train_metric.size))
                            k_align_te = int(min(dim, k_train, zeros_test_metric.size))

                            is_ablate_v = op["group"] == "ablate_V"
                            if is_ablate_v:
                                H_tr = np.asarray(H_base, dtype=_DTF, copy=True)
                                eig_tr = np.sort(np.linalg.eigvalsh(0.5 * (H_tr + H_tr.T))).astype(_DTF)
                                m_tr = smw(
                                    eig_tr,
                                    zeros_train_metric,
                                    k_align_tr,
                                    spacing_fn=v.spacing_mse_normalized,
                                    ks_fn=v.ks_against_wigner_gue,
                                    norm_gaps_fn=v.normalized_gaps,
                                )
                                od_tr = 0.0
                                J_tr = oos.pareto_objective_j(
                                    float(m_tr["spectral_log_mse"]),
                                    float(m_tr["spacing_mse_normalized"]),
                                    float(m_tr["ks_wigner"]),
                                    od_tr,
                                )
                                train_out = {
                                    "H_final": H_tr,
                                    "H_base": np.asarray(H_base, dtype=_DTF, copy=True),
                                    "V_diag_last": np.zeros((dim,), dtype=_DTF),
                                    "rows": [],
                                    "meta": {
                                        "converged_operator": True,
                                        "n_iter": 0,
                                        "eig_error": None,
                                        "note": "ablate_V_no_mixing",
                                    },
                                    "spectral_log_mse_train": float(m_tr["spectral_log_mse"]),
                                    "spacing_mse_normalized_train": float(m_tr["spacing_mse_normalized"]),
                                    "ks_wigner_train": float(m_tr["ks_wigner"]),
                                    "pareto_objective_train": J_tr,
                                    "operator_diff_final": od_tr,
                                    "final_row": {},
                                }
                            else:
                                train_out = oos.train_renormalized_with_variant(
                                    H_base=H_base,
                                    z_pool_sorted=z_sorted,
                                    train_ordinates=tr_ord,
                                    train_ordered=train_ordered,
                                    dim=dim,
                                    k_train=k_train,
                                    alpha=float(ALPHA),
                                    lambda_p_dim=lp,
                                    beta0=float(cfg["beta0"]),
                                    tau_beta=float(cfg["tau"]),
                                    beta_floor=float(cfg["beta_floor"]),
                                    smooth_sigma_dim=sm,
                                    clip_lo=clip_lo,
                                    clip_hi=clip_hi,
                                    diag_shift=float(DIAG_SHIFT),
                                    abs_cap_factor=float(ABS_CAP),
                                    zeros_train_metric=zeros_train_metric,
                                    spacing_fn=v.spacing_mse_normalized,
                                    ks_fn=v.ks_against_wigner_gue,
                                    norm_gaps_fn=v.normalized_gaps,
                                    max_iter=int(args.max_iter),
                                    tol=float(args.tol),
                                    variant=variant,
                                    freeze_v_after=int(freeze_v_after) if freeze_v_after is not None else None,
                                    target_blind=target_blind,
                                    on_train_iter=on_train_cb,
                                )

                            H_train = np.asarray(train_out["H_final"], dtype=_DTF, copy=True)
                            H_base_tr = np.asarray(train_out["H_base"], dtype=_DTF, copy=True)
                            V_train = np.asarray(train_out["V_diag_last"], dtype=_DTF, copy=True).reshape(-1)
                            meta_tr = train_out.get("meta") or {}
                            eig_err = meta_tr.get("eig_error")
                            finite = bool(np.isfinite(H_train).all())
                            sa = float(v13l1.self_adjointness_fro(H_train))
                            od_final = float(train_out.get("operator_diff_final", float("nan")))

                            acc_tr = meets_train_acceptance(
                                od=od_final,
                                sp=float(train_out["spacing_mse_normalized_train"]),
                                ks=float(train_out["ks_wigner_train"]),
                                finite=finite,
                                sa=sa,
                                eig_error=eig_err if isinstance(eig_err, str) else None,
                            )

                            J_train = float(train_out["pareto_objective_train"])

                            for fmode in freeze_modes:
                                m_te, od_te_raw = oos.eval_on_window(
                                    H=H_train,
                                    H_base=H_base_tr,
                                    z_window_sorted=zeros_test_metric,
                                    k_align=int(k_align_te),
                                    dim=dim,
                                    k_train=k_train,
                                    alpha=float(ALPHA),
                                    lambda_p_dim=lp,
                                    smooth_sigma_dim=sm,
                                    clip_lo=clip_lo,
                                    clip_hi=clip_hi,
                                    diag_shift=float(DIAG_SHIFT),
                                    abs_cap_factor=float(ABS_CAP),
                                    spacing_fn=v.spacing_mse_normalized,
                                    ks_fn=v.ks_against_wigner_gue,
                                    norm_gaps_fn=v.normalized_gaps,
                                    mode=str(fmode),
                                    V_fixed=V_train,
                                )
                                if fmode == "full_H":
                                    od_te = 0.0
                                else:
                                    od_te = float(od_te_raw) if math.isfinite(float(od_te_raw)) else 0.0

                                J_te = float(
                                    oos.pareto_objective_j(
                                        float(m_te["spectral_log_mse"]),
                                        float(m_te["spacing_mse_normalized"]),
                                        float(m_te["ks_wigner"]),
                                        od_te,
                                    )
                                )
                                gap_J = J_te - J_train
                                ratio_J = J_te / max(J_train, 1e-12)

                                acc_te = meets_test_acceptance(
                                    sp=float(m_te["spacing_mse_normalized"]),
                                    ks=float(m_te["ks_wigner"]),
                                    finite=finite,
                                    sa=sa,
                                    eig_error=eig_err if isinstance(eig_err, str) else None,
                                )

                                all_runs.append(
                                    {
                                        "dim": dim,
                                        "k_train": k_train,
                                        "target_group": target_group,
                                        "operator_group": op["group"],
                                        "operator_sub_id": op_sub,
                                        "potential_variant": pdef["id"],
                                        "potential_experimental": bool(pdef.get("experimental", False)),
                                        "freeze_mode": fmode,
                                        "window_note": win_note,
                                        "train_ordered": train_ordered,
                                        "lambda_p_effective": lp,
                                        "smooth_sigma_effective": sm,
                                        "clip_percentiles": json.dumps([clip_lo, clip_hi]),
                                        "word": word_json,
                                        "spectral_log_mse_train": float(train_out["spectral_log_mse_train"]),
                                        "spacing_mse_normalized_train": float(train_out["spacing_mse_normalized_train"]),
                                        "ks_wigner_train": float(train_out["ks_wigner_train"]),
                                        "pareto_objective_train": J_train,
                                        "spectral_log_mse_test": float(m_te["spectral_log_mse"]),
                                        "spacing_mse_normalized_test": float(m_te["spacing_mse_normalized"]),
                                        "ks_wigner_test": float(m_te["ks_wigner"]),
                                        "pareto_objective_test": J_te,
                                        "generalization_gap_J": gap_J,
                                        "generalization_ratio_J": ratio_J,
                                        "operator_diff_final": od_final,
                                        "operator_diff_test_mode": od_te,
                                        "self_adjointness_fro": sa,
                                        "finite": finite,
                                        "accepted_train": acc_tr,
                                        "accepted_test": acc_te,
                                        "eig_error": eig_err,
                                        "n_iter": int(meta_tr.get("n_iter", 0)),
                                        "converged_operator": bool(meta_tr.get("converged_operator", False)),
                                    }
                                )
                        except Exception as ex:
                            err_s = repr(ex)
                            print(f"[v13o1] WARNING job failed: {err_s}", flush=True)
                            append_failed_rows(
                                err_msg=err_s,
                                dim_=dim,
                                k_train_=k_train,
                                target_group_=target_group,
                                op=op,
                                op_sub_=op_sub,
                                pdef=pdef,
                                word_json_=word_json,
                                win_note_="",
                                train_ordered_=False,
                                lp_=lp,
                                sm_=sm,
                                clip_lo_=clip_lo,
                                clip_hi_=clip_hi,
                            )

    total_elapsed = time.perf_counter() - t0
    print(
        f"[v13o1] completed {job_idx}/{total_jobs} jobs "
        f"in {fmt_time(total_elapsed)} "
        f"avg/job={fmt_time(total_elapsed / max(job_idx, 1))}",
        flush=True,
    )

    # --- group summaries ---
    keys_grp = ("dim", "target_group", "operator_group", "potential_variant", "freeze_mode")
    group_map: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in all_runs:
        key = tuple(r[k] for k in keys_grp)
        group_map.setdefault(key, []).append(r)

    group_summary: List[Dict[str, Any]] = []
    for key, rs in sorted(group_map.items(), key=lambda kv: kv[0]):
        d, tg, og, pv, fm = key
        acc_t = float(sum(1 for x in rs if x.get("accepted_test"))) / max(1, len(rs))
        acc_tr = float(sum(1 for x in rs if x.get("accepted_train"))) / max(1, len(rs))
        Js = [float(x["pareto_objective_test"]) for x in rs if math.isfinite(float(x["pareto_objective_test"]))]
        group_summary.append(
            {
                "dim": d,
                "target_group": tg,
                "operator_group": og,
                "potential_variant": pv,
                "freeze_mode": fm,
                "n": len(rs),
                "accepted_test_rate": acc_t,
                "accepted_train_rate": acc_tr,
                "mean_J_test": float(np.mean(Js)) if Js else float("nan"),
                "min_J_test": float(np.min(Js)) if Js else float("nan"),
                "mean_gap_J": float(np.mean([float(x["generalization_gap_J"]) for x in rs if math.isfinite(float(x["generalization_gap_J"]))]))
                if rs
                else float("nan"),
            }
        )

    # effect sizes: primary full_V full_H real vs same-dim control operator means of J_test
    effect_rows: List[Dict[str, Any]] = []
    for dim in sorted({int(c["dim"]) for c in dim_cfgs}):
        ref = [
            r
            for r in all_runs
            if int(r["dim"]) == dim
            and r["target_group"] == "real_zeta"
            and r["operator_group"] == "primary_word_seed6"
            and r["potential_variant"] == "full_V"
            and r["freeze_mode"] == "full_H"
        ]
        if not ref:
            continue
        Jref = float(np.mean([float(r["pareto_objective_test"]) for r in ref if math.isfinite(float(r["pareto_objective_test"]))]))
        ogs = sorted({r["operator_group"] for r in all_runs if int(r["dim"]) == dim})
        for og in ogs:
            if og == "primary_word_seed6":
                continue
            crs = [
                r
                for r in all_runs
                if int(r["dim"]) == dim
                and r["target_group"] == "real_zeta"
                and r["operator_group"] == og
                and r["potential_variant"] == "full_V"
                and r["freeze_mode"] == "full_H"
            ]
            if not crs:
                continue
            mJ = float(np.mean([float(r["pareto_objective_test"]) for r in crs if math.isfinite(float(r["pareto_objective_test"]))]))
            effect_rows.append(
                {
                    "dim": dim,
                    "operator_group": og,
                    "delta_mean_J_test_vs_primary": mJ - Jref,
                    "mean_J_test_primary": Jref,
                    "mean_J_test_control": mJ,
                }
            )

    # best by dim: lowest J_test among accepted_test rows
    best_by_dim: List[Dict[str, Any]] = []
    for dim in sorted({int(c["dim"]) for c in dim_cfgs}):
        cand_rows = [r for r in all_runs if int(r["dim"]) == dim and r.get("accepted_test") and math.isfinite(float(r["pareto_objective_test"]))]
        if not cand_rows:
            best_by_dim.append({"dim": dim, "note": "no_accepted_test_rows"})
            continue
        best = min(cand_rows, key=lambda x: float(x["pareto_objective_test"]))
        best_by_dim.append(dict(best))

    cls, conclusion, next_step = classify_v13o1(all_runs, group_summary)

    primary_candidate = {
        "id": "seed_6",
        "word": list(PRIMARY_WORD),
        "description": "Primary braid word from V13 candidate theorem runs (computational candidate only).",
    }

    acceptance_block = {
        "train": {
            "operator_diff_final_max": 1e-3,
            "spacing_mse_normalized_max": 1.2,
            "ks_wigner_max": 0.25,
            "self_adjointness_fro_max": 1e-12,
            "finite_required": True,
        },
        "test": {
            "spacing_mse_normalized_max": 1.5,
            "ks_wigner_max": 0.30,
            "self_adjointness_fro_max": 1e-12,
            "finite_required": True,
        },
    }

    payload = {
        "warning": "Computational evidence only; not a proof of RH.",
        "status": "V13O.1 out-of-sample controls",
        "primary_candidate": primary_candidate,
        "dims": [int(c["dim"]) for c in dim_cfgs],
        "potential_variants": [p["id"] for p in pot_defs],
        "acceptance": acceptance_block,
        "inputs": {
            "candidate_json": str(_resolve(args.candidate_json)),
            "formula_json": str(_resolve(args.formula_json)),
            "v13n_summary": str(_resolve(args.v13n_summary)),
            "candidate_loaded": cand is not None,
            "formula_loaded": form is not None,
            "v13n_loaded": v13n is not None,
        },
        "runs": all_runs,
        "group_summary": group_summary,
        "effect_sizes": effect_rows,
        "classification": cls,
        "main_conclusion": conclusion,
        "next_step": next_step,
        "smoke_mode": bool(args.smoke),
        "seed": base_seed,
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "n_controls": int(args.n_controls),
        "n_random_words": int(args.n_random_words),
        "n_random_sym": int(args.n_random_sym),
        "progress_every": int(args.progress_every),
        "verbose_iter_every": int(args.verbose_iter_every),
    }

    def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in fieldnames})

    if all_runs:
        write_csv(out_dir / "v13o1_summary.csv", list(all_runs[0].keys()), all_runs)
    if group_summary:
        write_csv(out_dir / "v13o1_group_summary.csv", list(group_summary[0].keys()), group_summary)
    if effect_rows:
        write_csv(out_dir / "v13o1_effect_sizes.csv", list(effect_rows[0].keys()), effect_rows)
    if best_by_dim and isinstance(best_by_dim[0], dict) and best_by_dim[0].get("dim") is not None:
        # flatten best rows keys union
        bf = sorted({k for b in best_by_dim for k in b.keys()})
        write_csv(out_dir / "v13o1_best_by_dim.csv", bf, best_by_dim)

    with open(out_dir / "v13o1_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)

    # Markdown report (sections per user spec)
    md_parts = [
        "# V13O.1 Out-of-sample zeta-window validation\n\n",
        "> **Warning:** computational evidence for a **finite-dimensional approximant** only; **not** a proof of RH. ",
        "Do not interpret as a classical Hilbert–Pólya operator.\n\n",
        "## 1. Motivation\n\n",
        "V13O highlighted **specificity risk**: self-consistent potentials can fit in-window zeta structure while ",
        "GUE/Poisson or ablation controls sometimes score better under the same metrics. V13O.1 separates **train-window** ",
        "spectral fitting from **held-out test-window** behavior and adds **constrained-potential** variants.\n\n",
        "## 2. Train/test split\n\n",
        "For each dimension `d`, use `k_train` accepted V13N/V13M effective window sizes. ",
        "**Train** targets are the first `k_train` ordinates of a length-`2*k_train` block; **test** are the next `k_train`. ",
        "Blocks are drawn from sorted zeta zeros (or synthetic targets). See per-run `window_note`.\n\n",
        "## 3. Operator and potential variants\n\n",
        "- **Freeze modes on test:** `full_H` (frozen trained operator), `fixed_V` (kernel fixed, train diagonal potential), ",
        "`fixed_K_update_V_test` (fixed word kernel, one potential refresh against the **test** window without refitting the word).\n",
        "- **Potentials:** see `potential_variants` in JSON (`target_blind_V` is **experimental**).\n\n",
        "## 4. Controls\n\n",
        "Targets: real / shuffled / reversed / GUE / Poisson. Operators: primary word, random words, rejected word, ",
        "K/V/L ablations, random symmetric baseline.\n\n",
        "## 5. Results by dimension\n\n",
        f"Classification: **{cls}**. See `v13o1_summary.csv` and `v13o1_group_summary.csv`.\n\n",
        "## 6. Generalization gaps\n\n",
        "Column `generalization_gap_J` reports `J_test - J_train` using the Pareto objective with test-mode operator step where applicable.\n\n",
        "## 7. Specificity classification\n\n",
        f"{conclusion}\n\n",
        "## 8. Failure modes\n\n",
        "High test spacing MSE, high Wigner KS on the test window, or loss of self-adjointness/numerical finiteness indicate ",
        "out-of-sample breakdown or overfitting of the train window.\n\n",
        "## 9. Next steps\n\n",
        f"{next_step}\n\n",
        "## Files\n\n",
        "- `v13o1_results.json`, `v13o1_summary.csv`, `v13o1_group_summary.csv`, `v13o1_effect_sizes.csv`, `v13o1_best_by_dim.csv`\n\n",
        "## Run examples\n\n",
        "```text\n",
        run_examples,
        "```\n",
    ]
    (out_dir / "v13o1_report.md").write_text("".join(md_parts), encoding="utf-8")

    tex_body = (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\title{V13O.1 OOS controls (finite-dimensional candidate)}\n"
        "\\date{}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\section*{Disclaimer}\n"
        "Computational evidence only; not a proof of RH. Finite-dimensional approximant; "
        "not a classical Hilbert--P\\'olya operator.\n\n"
        "\\section{Classification}\n"
        + latex_escape(cls)
        + "\n\n\\section{Conclusion}\n"
        + latex_escape(conclusion)
        + "\n\n\\section{Next steps}\n"
        + latex_escape(next_step)
        + "\n\n\\end{document}\n"
    )
    (out_dir / "v13o1_report.tex").write_text(tex_body, encoding="utf-8")

    _pdf_exe = _find_pdflatex()
    if _pdf_exe is None:
        print("[v13o1] WARNING: pdflatex not found; skipping PDF.", flush=True)
    elif try_pdflatex(out_dir / "v13o1_report.tex", out_dir, "v13o1_report.pdf"):
        print(f"Wrote {out_dir / 'v13o1_report.pdf'}", flush=True)
    else:
        print("[v13o1] WARNING: pdflatex failed or did not produce v13o1_report.pdf.", flush=True)

    print(f"Wrote {out_dir / 'v13o1_results.json'} ({len(all_runs)} runs)")
    print(f"Classification: {cls}")


if __name__ == "__main__":
    main()
