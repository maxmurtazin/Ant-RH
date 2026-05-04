#!/usr/bin/env python3
"""
V13O: controls and ablation validation for the V13N renormalized DTES–Hilbert–Pólya operator family.

  python3 scripts/run_v13o_controls_ablation.py \\
    --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\
    --formula_json runs/v13_operator_formula/formula_components_summary.json \\
    --v13n_summary runs/v13n_theorem_report/v13n_summary.json \\
    --out_dir runs/v13o_controls_ablation \\
    --seed 42 --n_controls 30 --max_iter 300 --tol 1e-3
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
CLIP = (0.5, 99.5)
DIAG_SHIFT = 1e-6
ABS_CAP = 5.0

PRIMARY_WORD = [-4, -2, -4, -2, -2, -1, -1]
REJECTED_WORD = [6, 4, -1, 1, 1, 1, 4]

CONTROL_GROUPS = [
    "real_zeta_full",
    "shuffled_zeta",
    "reversed_zeta",
    "GUE_synthetic_targets",
    "Poisson_targets",
    "random_words",
    "rejected_word_seed17",
    "ablate_K",
    "ablate_V",
    "ablate_L",
    "random_symmetric_baseline",
]

DIM_CONFIGS: List[Dict[str, Any]] = [
    {
        "dim": 64,
        "word": list(PRIMARY_WORD),
        "k_eff": 45,
        "lambda_p": 2.121320343559643,
        "geo_sigma": 0.5045378491522287,
        "smooth_sigma": 0.7071067811865476,
        "beta0": 0.3,
        "tau": 200.0,
        "beta_floor": 0.03,
        "source": "v13m1",
    },
    {
        "dim": 128,
        "word": list(PRIMARY_WORD),
        "k_eff": 96,
        "lambda_p": 3.0,
        "geo_sigma": 0.6,
        "smooth_sigma": 1.0,
        "beta0": 0.3,
        "tau": 500.0,
        "beta_floor": 0.03,
        "source": "v13m2",
    },
    {
        "dim": 256,
        "word": list(PRIMARY_WORD),
        "k_eff": 128,
        "lambda_p": 2.121320343559643,
        "geo_sigma": 0.5045378491522287,
        "smooth_sigma": 1.25,
        "beta0": 0.3,
        "tau": 300.0,
        "beta_floor": 0.03,
        "source": "v13m2",
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


def condition_proxy(eig: np.ndarray) -> float:
    e = np.sort(np.asarray(eig, dtype=_DTF).reshape(-1))
    e = e[np.isfinite(e)]
    if e.size < 2:
        return float("nan")
    sp = np.diff(e)
    sp = sp[np.isfinite(sp)]
    if sp.size == 0:
        return float("nan")
    med = float(np.median(sp))
    return float(abs(float(e[-1]) - float(e[0])) / max(1e-12, med))


def operator_hash(H: np.ndarray) -> str:
    b = np.asarray(H, dtype=np.float64).tobytes()
    return hashlib.sha256(b).hexdigest()[:24]


def random_art_word(length: int, rng: np.random.Generator) -> List[int]:
    alphabet = list(range(-6, 0)) + list(range(1, 7))
    return [int(rng.choice(alphabet)) for _ in range(int(length))]


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


def pareto_J(sl: float, sp: float, ks: float, od: Optional[float]) -> float:
    from core.v13l2_pareto import pareto_objective_J

    if od is None:
        return float(sl) + 0.05 * float(sp) + float(ks)
    try:
        odv = float(od)
    except (TypeError, ValueError):
        return float(sl) + 0.05 * float(sp) + float(ks)
    if not math.isfinite(odv):
        return float(sl) + 0.05 * float(sp) + float(ks)
    return float(pareto_objective_J(sl, sp, ks, odv))


def meets_acceptance(
    *,
    od: Optional[float],
    sp: float,
    ks: float,
    finite: bool,
    sa: float,
    eig_error: Optional[str],
) -> bool:
    if eig_error is not None:
        return False
    if od is None or not math.isfinite(od) or od > 1e-3:
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


def eval_pass_no_fp(sp: float, ks: float, finite: bool, sa: float) -> bool:
    return (
        finite
        and math.isfinite(sa)
        and sa <= 1e-12
        and math.isfinite(sp)
        and sp <= 1.2
        and math.isfinite(ks)
        and ks <= 0.25
    )


def metrics_on_H(
    H: np.ndarray,
    z_metric: np.ndarray,
    k_align: int,
    v: Any,
    m1: Any,
) -> Tuple[Dict[str, float], np.ndarray]:
    H = np.asarray(H, dtype=_DTF, copy=False)
    eig = np.sort(np.linalg.eigvalsh(0.5 * (H + H.T)).astype(_DTF))
    m = m1.spectral_metrics_window(
        eig,
        z_metric,
        int(k_align),
        spacing_fn=v.spacing_mse_normalized,
        ks_fn=v.ks_against_wigner_gue,
        norm_gaps_fn=v.normalized_gaps,
    )
    return m, eig


def eig_spacing_stats(eig: np.ndarray) -> Tuple[float, float, float]:
    e = np.sort(np.asarray(eig, dtype=_DTF).reshape(-1))
    e = e[np.isfinite(e)]
    if e.size < 2:
        return float("nan"), float("nan"), float("nan")
    sp = np.diff(e)
    return float(np.mean(sp)), float(np.std(sp)), float(e[-1] - e[0])


def run_iterative_case(
    *,
    H_base: np.ndarray,
    z_pool: np.ndarray,
    k_eff: int,
    dim: int,
    target_ordered: Optional[np.ndarray],
    beta0: float,
    tau: float,
    beta_floor: float,
    smooth_sigma: float,
    lambda_p: float,
    v: Any,
    m1: Any,
    max_iter: int,
    tol: float,
) -> Dict[str, Any]:
    z_sorted = np.asarray(z_pool, dtype=_DTF).reshape(-1)
    z_sorted = z_sorted[np.isfinite(z_sorted) & (z_sorted > 0.0)]
    z_sorted = np.sort(z_sorted)
    k_align = int(min(int(dim), int(k_eff), int(z_sorted.size)))
    zeros_metric = z_sorted[: max(k_align, 1)].astype(_DTF, copy=False)

    out = m1.run_renormalized_cell(
        H_base=H_base,
        z_pool_positive=z_sorted,
        dim=int(dim),
        zeros_eff=int(k_eff),
        alpha=float(ALPHA),
        lambda_p_dim=float(lambda_p),
        beta0=float(beta0),
        tau_beta=float(tau),
        beta_floor=float(beta_floor),
        smooth_sigma_dim=float(smooth_sigma),
        clip_percentiles=(float(CLIP[0]), float(CLIP[1])),
        diag_shift=float(DIAG_SHIFT),
        abs_cap_factor=float(ABS_CAP),
        zeros_true_for_metrics=zeros_metric,
        k_align=int(k_align),
        spacing_fn=v.spacing_mse_normalized,
        ks_fn=v.ks_against_wigner_gue,
        norm_gaps_fn=v.normalized_gaps,
        stagnation_stop_only_if_operator_diff_below=1e-3,
        target_positive_ordered=target_ordered,
        max_iter=int(max_iter),
        tol=float(tol),
    )
    return out


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


def summarize_rows(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    xs = [float(r[key]) for r in rows if key in r and r[key] is not None]
    xs = [x for x in xs if math.isfinite(x)]
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan")}
    a = np.asarray(xs, dtype=np.float64)
    return {"mean": float(np.mean(a)), "std": float(np.std(a)), "min": float(np.min(a))}


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O controls and ablations for V13N operator family.")
    ap.add_argument("--candidate_json", type=str, default="runs/v13_candidate_theorem/candidate_operators.json")
    ap.add_argument("--formula_json", type=str, default="runs/v13_operator_formula/formula_components_summary.json")
    ap.add_argument("--v13n_summary", type=str, default="runs/v13n_theorem_report/v13n_summary.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13o_controls_ablation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_controls", type=int, default=30)
    ap.add_argument("--max_iter", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import importlib.util

    def _load_v13_validate() -> Any:
        path = Path(ROOT) / "scripts" / "validate_candidates_v13.py"
        spec = importlib.util.spec_from_file_location("_v13_validate", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_v13_validate"] = mod
        spec.loader.exec_module(mod)
        return mod

    v = _load_v13_validate()
    from core import v13l_self_consistent as v13l
    from core import v13l1_stabilized as v13l1
    from core import v13m1_renormalized as m1
    from core.artin_operator import sample_domain

    _load_json(_resolve(args.candidate_json))
    _load_json(_resolve(args.formula_json))
    _load_json(_resolve(args.v13n_summary))

    z_pool = v._load_zeros(512)
    z_sorted = np.asarray(z_pool, dtype=_DTF).reshape(-1)
    z_sorted = z_sorted[np.isfinite(z_sorted) & (z_sorted > 0.0)]
    z_sorted = np.sort(z_sorted)

    n_ctrl = max(1, int(args.n_controls))
    base_seed = int(args.seed)
    all_runs: List[Dict[str, Any]] = []

    for cfg in DIM_CONFIGS:
        dim = int(cfg["dim"])
        k_eff = int(cfg["k_eff"])
        rng = np.random.default_rng(base_seed + dim * 10007)

        Z = sample_domain(dim, seed=base_seed)
        word0 = [int(x) for x in cfg["word"]]
        H_full = build_h(
            z_points=Z,
            word=word0,
            v13l=v13l,
            geo_sigma=float(cfg["geo_sigma"]),
            laplacian_weight=1.0,
            geo_weight=float(GEO_WEIGHT),
        )

        def append_row(
            *,
            control_group: str,
            control_id: str,
            word: List[int],
            H_op: np.ndarray,
            out_loop: Optional[Dict[str, Any]],
            no_fp: bool,
            eig_error: Optional[str],
        ) -> None:
            meta_loop = (out_loop or {}).get("meta") or {}
            rows_loop = list((out_loop or {}).get("rows") or [])
            eff_err = eig_error if eig_error is not None else meta_loop.get("eig_error")

            if no_fp:
                k_al = int(min(dim, k_eff, z_sorted.size))
                zm = z_sorted[: max(k_al, 1)]
                m, _e = metrics_on_H(H_op, zm, k_al, v, m1)
                sl = float(m["spectral_log_mse"])
                sp = float(m["spacing_mse_normalized"])
                ks = float(m["ks_wigner"])
                od = None
                n_iter = 0
                conv = False
                H_fin = np.asarray(H_op, dtype=_DTF, copy=True)
            elif rows_loop:
                rN = rows_loop[-1]
                sl = float(rN.get("spectral_log_mse", float("nan")))
                sp = float(rN.get("spacing_mse_normalized", float("nan")))
                ks = float(rN.get("ks_wigner", float("nan")))
                od = float(rN.get("operator_diff", float("nan")))
                n_iter = int(meta_loop.get("n_iter", len(rows_loop)))
                conv = bool(meta_loop.get("converged_operator", False))
                H_fin = np.asarray((out_loop or {}).get("H_final", H_op), dtype=_DTF, copy=True)
            elif out_loop is not None:
                H_fin = np.asarray(out_loop.get("H_final", H_op), dtype=_DTF, copy=True)
                k_al = int(min(dim, k_eff, z_sorted.size))
                zm = z_sorted[: max(k_al, 1)]
                m, _e = metrics_on_H(H_fin, zm, k_al, v, m1)
                sl = float(m["spectral_log_mse"])
                sp = float(m["spacing_mse_normalized"])
                ks = float(m["ks_wigner"])
                fod = out_loop.get("final_operator_diff")
                od = float(fod) if fod is not None and math.isfinite(float(fod)) else float("nan")
                n_iter = int(meta_loop.get("n_iter", 0))
                conv = bool(meta_loop.get("converged_operator", False))
            else:
                sl = sp = ks = float("nan")
                od = float("nan")
                n_iter = 0
                conv = False
                H_fin = np.asarray(H_op, dtype=_DTF, copy=True)

            finite = bool(np.isfinite(H_fin).all())
            sa = float(v13l1.self_adjointness_fro(H_fin))
            eig = np.sort(np.linalg.eigvalsh(0.5 * (H_fin + H_fin.T)).astype(_DTF))
            sp_m, sp_s, spec_r = eig_spacing_stats(eig)
            cp = condition_proxy(eig)
            J = pareto_J(sl, sp, ks, od)
            acc = meets_acceptance(od=od, sp=sp, ks=ks, finite=finite, sa=sa, eig_error=eff_err)
            ev_no_fp = eval_pass_no_fp(sp, ks, finite, sa)

            all_runs.append(
                {
                    "dim": dim,
                    "k_eff": k_eff,
                    "control_group": control_group,
                    "control_id": control_id,
                    "word": json.dumps(word),
                    "accepted": acc,
                    "eval_pass_without_fixed_point": ev_no_fp,
                    "finite": finite,
                    "self_adjointness_fro": sa,
                    "operator_diff_final": od,
                    "spectral_log_mse": sl,
                    "spacing_mse_normalized": sp,
                    "ks_wigner": ks,
                    "pareto_objective": J,
                    "eig_min": float(eig[0]) if eig.size else float("nan"),
                    "eig_max": float(eig[-1]) if eig.size else float("nan"),
                    "eig_spacing_mean": sp_m,
                    "eig_spacing_std": sp_s,
                    "spectral_range": spec_r,
                    "condition_proxy": cp,
                    "n_iter": n_iter,
                    "converged_operator": conv,
                    "eig_error": eff_err,
                    "operator_hash": operator_hash(H_fin),
                }
            )

        # A real_zeta_full
        out_a = run_iterative_case(
            H_base=H_full,
            z_pool=z_sorted,
            k_eff=k_eff,
            dim=dim,
            target_ordered=None,
            beta0=float(cfg["beta0"]),
            tau=float(cfg["tau"]),
            beta_floor=float(cfg["beta_floor"]),
            smooth_sigma=float(cfg["smooth_sigma"]),
            lambda_p=float(cfg["lambda_p"]),
            v=v,
            m1=m1,
            max_iter=int(args.max_iter),
            tol=float(args.tol),
        )
        append_row(
            control_group="real_zeta_full",
            control_id="real",
            word=word0,
            H_op=H_full,
            out_loop=out_a,
            no_fp=False,
            eig_error=(out_a.get("meta") or {}).get("eig_error"),
        )

        z_base = z_sorted[:k_eff].astype(_DTF, copy=True)

        # B shuffled
        for i in range(n_ctrl):
            r2 = np.random.default_rng(base_seed + dim * 100003 + i)
            perm = z_base.copy()
            r2.shuffle(perm)
            out_b = run_iterative_case(
                H_base=H_full,
                z_pool=z_sorted,
                k_eff=k_eff,
                dim=dim,
                target_ordered=perm,
                beta0=float(cfg["beta0"]),
                tau=float(cfg["tau"]),
                beta_floor=float(cfg["beta_floor"]),
                smooth_sigma=float(cfg["smooth_sigma"]),
                lambda_p=float(cfg["lambda_p"]),
                v=v,
                m1=m1,
                max_iter=int(args.max_iter),
                tol=float(args.tol),
            )
            append_row(
                control_group="shuffled_zeta",
                control_id=f"shuffle_{i}",
                word=word0,
                H_op=H_full,
                out_loop=out_b,
                no_fp=False,
                eig_error=(out_b.get("meta") or {}).get("eig_error"),
            )

        # C reversed
        rev = z_base[::-1].copy()
        out_c = run_iterative_case(
            H_base=H_full,
            z_pool=z_sorted,
            k_eff=k_eff,
            dim=dim,
            target_ordered=rev,
            beta0=float(cfg["beta0"]),
            tau=float(cfg["tau"]),
            beta_floor=float(cfg["beta_floor"]),
            smooth_sigma=float(cfg["smooth_sigma"]),
            lambda_p=float(cfg["lambda_p"]),
            v=v,
            m1=m1,
            max_iter=int(args.max_iter),
            tol=float(args.tol),
        )
        append_row(
            control_group="reversed_zeta",
            control_id="reversed",
            word=word0,
            H_op=H_full,
            out_loop=out_c,
            no_fp=False,
            eig_error=(out_c.get("meta") or {}).get("eig_error"),
        )

        # D GUE
        for i in range(n_ctrl):
            gue = gue_positive_ordinates(k_eff, np.random.default_rng(base_seed + dim * 200017 + i))
            out_d = run_iterative_case(
                H_base=H_full,
                z_pool=z_sorted,
                k_eff=k_eff,
                dim=dim,
                target_ordered=gue.astype(_DTF, copy=False),
                beta0=float(cfg["beta0"]),
                tau=float(cfg["tau"]),
                beta_floor=float(cfg["beta_floor"]),
                smooth_sigma=float(cfg["smooth_sigma"]),
                lambda_p=float(cfg["lambda_p"]),
                v=v,
                m1=m1,
                max_iter=int(args.max_iter),
                tol=float(args.tol),
            )
            append_row(
                control_group="GUE_synthetic_targets",
                control_id=f"gue_{i}",
                word=word0,
                H_op=H_full,
                out_loop=out_d,
                no_fp=False,
                eig_error=(out_d.get("meta") or {}).get("eig_error"),
            )

        # E Poisson
        for i in range(n_ctrl):
            poi = poisson_positive_ordinates(k_eff, np.random.default_rng(base_seed + dim * 300029 + i))
            out_e = run_iterative_case(
                H_base=H_full,
                z_pool=z_sorted,
                k_eff=k_eff,
                dim=dim,
                target_ordered=poi.astype(_DTF, copy=False),
                beta0=float(cfg["beta0"]),
                tau=float(cfg["tau"]),
                beta_floor=float(cfg["beta_floor"]),
                smooth_sigma=float(cfg["smooth_sigma"]),
                lambda_p=float(cfg["lambda_p"]),
                v=v,
                m1=m1,
                max_iter=int(args.max_iter),
                tol=float(args.tol),
            )
            append_row(
                control_group="Poisson_targets",
                control_id=f"poisson_{i}",
                word=word0,
                H_op=H_full,
                out_loop=out_e,
                no_fp=False,
                eig_error=(out_e.get("meta") or {}).get("eig_error"),
            )

        # F random words
        for i in range(n_ctrl):
            rw = random_art_word(len(word0), np.random.default_rng(base_seed + dim * 400043 + i))
            H_rw = build_h(
                z_points=Z,
                word=rw,
                v13l=v13l,
                geo_sigma=float(cfg["geo_sigma"]),
                laplacian_weight=1.0,
                geo_weight=float(GEO_WEIGHT),
            )
            out_f = run_iterative_case(
                H_base=H_rw,
                z_pool=z_sorted,
                k_eff=k_eff,
                dim=dim,
                target_ordered=None,
                beta0=float(cfg["beta0"]),
                tau=float(cfg["tau"]),
                beta_floor=float(cfg["beta_floor"]),
                smooth_sigma=float(cfg["smooth_sigma"]),
                lambda_p=float(cfg["lambda_p"]),
                v=v,
                m1=m1,
                max_iter=int(args.max_iter),
                tol=float(args.tol),
            )
            append_row(
                control_group="random_words",
                control_id=f"randword_{i}",
                word=rw,
                H_op=H_rw,
                out_loop=out_f,
                no_fp=False,
                eig_error=(out_f.get("meta") or {}).get("eig_error"),
            )

        # G rejected
        H_rej = build_h(
            z_points=Z,
            word=list(REJECTED_WORD),
            v13l=v13l,
            geo_sigma=float(cfg["geo_sigma"]),
            laplacian_weight=1.0,
            geo_weight=float(GEO_WEIGHT),
        )
        out_g = run_iterative_case(
            H_base=H_rej,
            z_pool=z_sorted,
            k_eff=k_eff,
            dim=dim,
            target_ordered=None,
            beta0=float(cfg["beta0"]),
            tau=float(cfg["tau"]),
            beta_floor=float(cfg["beta_floor"]),
            smooth_sigma=float(cfg["smooth_sigma"]),
            lambda_p=float(cfg["lambda_p"]),
            v=v,
            m1=m1,
            max_iter=int(args.max_iter),
            tol=float(args.tol),
        )
        append_row(
            control_group="rejected_word_seed17",
            control_id="seed17_like",
            word=list(REJECTED_WORD),
            H_op=H_rej,
            out_loop=out_g,
            no_fp=False,
            eig_error=(out_g.get("meta") or {}).get("eig_error"),
        )

        # H ablate K
        H_noK = build_h(
            z_points=Z,
            word=word0,
            v13l=v13l,
            geo_sigma=float(cfg["geo_sigma"]),
            laplacian_weight=1.0,
            geo_weight=0.0,
        )
        out_h = run_iterative_case(
            H_base=H_noK,
            z_pool=z_sorted,
            k_eff=k_eff,
            dim=dim,
            target_ordered=None,
            beta0=float(cfg["beta0"]),
            tau=float(cfg["tau"]),
            beta_floor=float(cfg["beta_floor"]),
            smooth_sigma=float(cfg["smooth_sigma"]),
            lambda_p=float(cfg["lambda_p"]),
            v=v,
            m1=m1,
            max_iter=int(args.max_iter),
            tol=float(args.tol),
        )
        append_row(
            control_group="ablate_K",
            control_id="geo_weight_0",
            word=word0,
            H_op=H_noK,
            out_loop=out_h,
            no_fp=False,
            eig_error=(out_h.get("meta") or {}).get("eig_error"),
        )

        # I ablate V
        append_row(
            control_group="ablate_V",
            control_id="lambda_p_0_eval",
            word=word0,
            H_op=H_full,
            out_loop=None,
            no_fp=True,
            eig_error=None,
        )

        # J ablate L
        H_noL = build_h(
            z_points=Z,
            word=word0,
            v13l=v13l,
            geo_sigma=float(cfg["geo_sigma"]),
            laplacian_weight=0.0,
            geo_weight=float(GEO_WEIGHT),
        )
        out_j = run_iterative_case(
            H_base=H_noL,
            z_pool=z_sorted,
            k_eff=k_eff,
            dim=dim,
            target_ordered=None,
            beta0=float(cfg["beta0"]),
            tau=float(cfg["tau"]),
            beta_floor=float(cfg["beta_floor"]),
            smooth_sigma=float(cfg["smooth_sigma"]),
            lambda_p=float(cfg["lambda_p"]),
            v=v,
            m1=m1,
            max_iter=int(args.max_iter),
            tol=float(args.tol),
        )
        append_row(
            control_group="ablate_L",
            control_id="laplacian_0",
            word=word0,
            H_op=H_noL,
            out_loop=out_j,
            no_fp=False,
            eig_error=(out_j.get("meta") or {}).get("eig_error"),
        )

        # K random symmetric baseline
        fro_t = float(np.linalg.norm(H_full, ord="fro"))
        for i in range(n_ctrl):
            r = rng.standard_normal((dim, dim))
            S = 0.5 * (r + r.T)
            fs = float(np.linalg.norm(S, ord="fro"))
            S = S * (fro_t / max(fs, 1e-12))
            S = S.astype(_DTF, copy=False)
            append_row(
                control_group="random_symmetric_baseline",
                control_id=f"randsym_{i}",
                word=word0,
                H_op=S,
                out_loop=None,
                no_fp=True,
                eig_error=None,
            )

    # --- summaries ---
    def rows_filter(dim_: int, grp: str) -> List[Dict[str, Any]]:
        return [r for r in all_runs if int(r["dim"]) == dim_ and r["control_group"] == grp]

    summary_rows: List[Dict[str, Any]] = []
    for dim in (64, 128, 256):
        for grp in CONTROL_GROUPS:
            rs = rows_filter(dim, grp)
            n = len(rs)
            acc_c = sum(1 for r in rs if r.get("accepted"))
            st_sp = summarize_rows(rs, "spacing_mse_normalized")
            st_ks = summarize_rows(rs, "ks_wigner")
            st_sl = summarize_rows(rs, "spectral_log_mse")
            st_J = summarize_rows(rs, "pareto_objective")
            ods = [
                float(r["operator_diff_final"])
                for r in rs
                if r.get("operator_diff_final") is not None and math.isfinite(float(r["operator_diff_final"]))
            ]
            st_od = (
                {"mean": float(np.mean(ods)), "std": float(np.std(ods)), "min": float(np.min(ods))}
                if ods
                else {"mean": float("nan"), "std": float("nan"), "min": float("nan")}
            )
            best_J = min((float(r["pareto_objective"]) for r in rs if math.isfinite(float(r["pareto_objective"]))), default=float("nan"))
            summary_rows.append(
                {
                    "dim": dim,
                    "control_group": grp,
                    "n": n,
                    "accepted_count": acc_c,
                    "accepted_rate": float(acc_c) / max(1, n),
                    "mean_spacing": st_sp["mean"],
                    "std_spacing": st_sp["std"],
                    "min_spacing": st_sp["min"],
                    "mean_ks": st_ks["mean"],
                    "std_ks": st_ks["std"],
                    "min_ks": st_ks["min"],
                    "mean_spectral_log_mse": st_sl["mean"],
                    "std_spectral_log_mse": st_sl["std"],
                    "min_spectral_log_mse": st_sl["min"],
                    "mean_pareto_objective": st_J["mean"],
                    "std_pareto_objective": st_J["std"],
                    "min_pareto_objective": st_J["min"],
                    "mean_operator_diff_final": st_od["mean"],
                    "std_operator_diff_final": st_od["std"],
                    "min_operator_diff_final": st_od["min"],
                    "best_J": best_J,
                }
            )

    # effect sizes vs real
    effect_rows: List[Dict[str, Any]] = []
    for dim in (64, 128, 256):
        real_rs = rows_filter(dim, "real_zeta_full")
        if not real_rs:
            continue
        r0 = real_rs[0]
        rJ = float(r0["pareto_objective"])
        rsp = float(r0["spacing_mse_normalized"])
        rks = float(r0["ks_wigner"])
        for grp in CONTROL_GROUPS:
            if grp == "real_zeta_full":
                continue
            crs = rows_filter(dim, grp)
            if not crs:
                continue
            mJ = float(np.mean([float(x["pareto_objective"]) for x in crs if math.isfinite(float(x["pareto_objective"]))])) if crs else float("nan")
            msp = float(np.mean([float(x["spacing_mse_normalized"]) for x in crs if math.isfinite(float(x["spacing_mse_normalized"]))])) if crs else float("nan")
            mks = float(np.mean([float(x["ks_wigner"]) for x in crs if math.isfinite(float(x["ks_wigner"]))])) if crs else float("nan")
            wins = sum(
                1
                for x in crs
                if math.isfinite(float(x["pareto_objective"])) and math.isfinite(rJ) and float(x["pareto_objective"]) > rJ
            )
            effect_rows.append(
                {
                    "dim": dim,
                    "control_group": grp,
                    "delta_mean_spacing": msp - rsp,
                    "delta_mean_ks": mks - rks,
                    "delta_mean_J": mJ - rJ,
                    "win_rate_real_better": float(wins) / max(1, len(crs)),
                }
            )

    # rank real by J within each dim (1 = best / lowest J)
    best_by_dim: List[Dict[str, Any]] = []
    ranks: Dict[int, int] = {}
    for dim in (64, 128, 256):
        dim_rows = [r for r in all_runs if int(r["dim"]) == dim and math.isfinite(float(r["pareto_objective"]))]
        dim_rows_sorted = sorted(dim_rows, key=lambda x: float(x["pareto_objective"]))
        real_r = [
            r
            for r in all_runs
            if int(r["dim"]) == dim and r["control_group"] == "real_zeta_full" and r.get("control_id") == "real"
        ]
        if real_r:
            rr = real_r[0]
            rk = next((i + 1 for i, row in enumerate(dim_rows_sorted) if row is rr), len(dim_rows_sorted) + 1)
            ranks[dim] = int(rk)
            br = dict(rr)
            br["real_rank_by_J"] = ranks[dim]
            br["n_total_runs_dim"] = len(dim_rows)
            best_by_dim.append(br)

    failed = [r for r in all_runs if r.get("eig_error") is not None or not r.get("finite", False)]

    # classification
    real_ok = [
        r
        for r in all_runs
        if r["control_group"] == "real_zeta_full" and r.get("control_id") == "real" and r.get("accepted")
    ]
    real_dims = sorted({int(r["dim"]) for r in real_ok})
    shuffle_rates = [summarize_accept_rate(all_runs, d, "shuffled_zeta") for d in (64, 128, 256)]
    randword_rates = [summarize_accept_rate(all_runs, d, "random_words") for d in (64, 128, 256)]
    abK = [summarize_accept_rate(all_runs, d, "ablate_K") for d in (64, 128, 256)]

    def mean(xs: List[float]) -> float:
        xs = [x for x in xs if math.isfinite(x)]
        return float(np.mean(xs)) if xs else float("nan")

    n_runs_dim = len([x for x in all_runs if int(x["dim"]) == 64]) if all_runs else 0
    top5_cut = max(1, int(math.ceil(0.05 * max(1, n_runs_dim))))

    classification = "weak_specificity"
    main_conclusion = "Real configuration is not clearly separated from several controls."
    next_step = "revise operator because effect is not specific"

    if len(real_dims) < 3:
        classification = "failed"
        main_conclusion = "real_zeta_full did not meet acceptance on all tested dimensions (64,128,256)."
        next_step = "debug reproducibility"
    else:
        m_sh = mean(shuffle_rates)
        m_rw = mean(randword_rates)
        m_abk = mean(abK)
        rank_ok = all(ranks.get(d, 999) <= top5_cut for d in (64, 128, 256))
        if m_sh < 0.05 and m_rw < 0.1 and m_abk < 0.05 and rank_ok:
            classification = "strong_specificity"
            main_conclusion = (
                "Real zeta + primary word is accepted on all dims while shuffle / random-word / ablate_K "
                "controls largely fail and real ranks in the top 5% by J."
            )
            next_step = "V13P analytic renormalization law + spectral triple formalization"
        elif m_sh > 0.25 and m_rw > 0.25:
            classification = "weak_specificity"
            main_conclusion = "Many randomized controls achieve similar acceptance or metrics to the real configuration."
            next_step = "revise operator because effect is not specific"
        else:
            classification = "medium_specificity"
            main_conclusion = "Real is accepted on all dims but some controls partially match metrics or acceptance."
            next_step = "V13O.1 stronger controls / larger n_controls / dim=512"

    control_rates: Dict[str, Any] = {}
    for d in (64, 128, 256):
        for grp in CONTROL_GROUPS:
            control_rates[f"dim{d}_{grp}"] = summarize_accept_rate(all_runs, d, grp)

    payload = {
        "classification": classification,
        "real_accepted_dims": real_dims,
        "real_rank_by_J": ranks,
        "control_accepted_rates": control_rates,
        "main_conclusion": main_conclusion,
        "next_step": next_step,
        "warning": "Computational falsification study only; not a proof of RH.",
        "n_runs": len(all_runs),
    }

    def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in fieldnames})

    run_fields = list(all_runs[0].keys()) if all_runs else []
    write_csv(out_dir / "v13o_all_runs.csv", run_fields, all_runs)

    sg_fields = list(summary_rows[0].keys()) if summary_rows else []
    write_csv(out_dir / "v13o_summary_by_group.csv", sg_fields, summary_rows)

    ef_fields = list(effect_rows[0].keys()) if effect_rows else []
    write_csv(out_dir / "v13o_effect_sizes.csv", ef_fields, effect_rows)

    bb_fields = list(best_by_dim[0].keys()) if best_by_dim else []
    write_csv(out_dir / "v13o_best_by_dim.csv", bb_fields, best_by_dim)

    ff_fields = list(failed[0].keys()) if failed else run_fields
    write_csv(out_dir / "v13o_failed_controls.csv", ff_fields, failed)

    with open(out_dir / "v13o_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **payload,
                "summary_by_group": summary_rows,
                "effect_sizes": effect_rows,
            },
            f,
            indent=2,
            allow_nan=True,
        )

    # Report md
    md = [
        "# V13O Controls and Ablations for the Renormalized DTES–Hilbert–Pólya Operator Family\n\n",
        "## 1. Warning\n\n",
        "> This is a **computational falsification / robustness** study, **not** a proof of the Riemann Hypothesis.\n\n",
        "## 2. Recap of real operator\n\n",
        r"$$H_d^\ast = \mathrm{Sym}\bigl(L_d + 10 K_{w,d} + \lambda_p(d) V_d(H_d^\ast)\bigr) + \varepsilon I$$" "\n\n",
        f"Primary word $w = {PRIMARY_WORD}$.\n\n",
        "## 3. Control summary table\n\n",
        "| control_group | dim | n | accepted_rate | mean spacing | mean KS | mean J | best J |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary_rows:
        md.append(
            f"| {row.get('control_group')} | {row.get('dim')} | {row.get('n')} | "
            f"{row.get('accepted_rate'):.4f} | {row.get('mean_spacing')} | {row.get('mean_ks')} | "
            f"{row.get('mean_pareto_objective')} | {row.get('best_J')} |\n"
        )
    md.append("\n## 4. Main falsification question\n\n")
    md.append(f"- **Classification:** `{classification}`\n")
    md.append(f"- **Real accepted dimensions:** {real_dims}\n")
    md.append(f"- **Real rank by J (1=best):** {json.dumps(ranks)}\n")
    md.append(f"- **Mean shuffle acceptance rate (across dims):** {mean(shuffle_rates):.4f}\n")
    md.append(f"- **Mean random_words acceptance rate:** {mean(randword_rates):.4f}\n")
    md.append(f"- **Ablate_K acceptance rate per dim:** {abK}\n\n")
    md.append("## 5. Interpretation\n\n")
    md.append(
        f"Outcome bucket: **{classification}**. See `v13o_results.json` for `main_conclusion` and `next_step`.\n\n"
    )
    md.append("## 6. Files\n\n")
    md.append("- `v13o_results.json`, `v13o_all_runs.csv`, `v13o_summary_by_group.csv`, `v13o_effect_sizes.csv`\n")
    md.append("- `v13o_best_by_dim.csv`, `v13o_failed_controls.csv`\n")
    (out_dir / "v13o_report.md").write_text("".join(md), encoding="utf-8")

    tex = [
        "\\documentclass[11pt]{article}\n",
        "\\usepackage[T1]{fontenc}\n",
        "\\usepackage[utf8]{inputenc}\n",
        "\\usepackage[margin=1in]{geometry}\n",
        "\\title{V13O Controls and Ablations}\n\\date{}\n\\begin{document}\n\\maketitle\n",
        "\\section*{Warning}\n",
        "Computational falsification study only; not a proof of RH.\n\n",
        "\\section{Classification}\n",
        latex_escape(classification) + "\n\n",
        "\\end{document}\n",
    ]
    (out_dir / "v13o_report.tex").write_text("".join(tex), encoding="utf-8")
    if try_pdflatex(out_dir / "v13o_report.tex", out_dir, "v13o_report.pdf"):
        print(f"Wrote {out_dir / 'v13o_report.pdf'}")
    else:
        print("PDF skipped.", flush=True)

    print(f"Wrote {out_dir / 'v13o_results.json'} ({len(all_runs)} runs)")
    print(f"Classification: {classification}")


def summarize_accept_rate(all_runs: List[Dict[str, Any]], dim: int, grp: str) -> float:
    rs = [r for r in all_runs if int(r["dim"]) == dim and r["control_group"] == grp]
    if not rs:
        return float("nan")
    return float(sum(1 for r in rs if r.get("accepted"))) / float(len(rs))


if __name__ == "__main__":
    main()
