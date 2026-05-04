#!/usr/bin/env python3
"""
V13I: control experiments for V13H candidates (random words, shuffled zeros, sign flips, permutations, GUE baseline).

  python3 scripts/run_v13_controls.py \\
    --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\
    --out_dir runs/v13_controls \\
    --zeros 128 --dim 128 --n_random 100 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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

if not os.environ.get("MPLCONFIGDIR"):
    _mpl_cfg = Path(ROOT) / ".mpl_cache"
    try:
        _mpl_cfg.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_cfg)
    except OSError:
        pass

_DTF = np.float64

METRIC_KEYS = [
    "spectral_log_mse",
    "spectral_raw_mse",
    "spacing_mse_normalized",
    "ks_wigner",
    "ramsey_score",
    "nijenhuis_defect",
    "comm_norm_proxy",
    "operator_hash",
    "eig_min",
    "eig_max",
    "n_eig",
    "n_zeros",
]


def _load_v13_validate() -> Any:
    path = Path(ROOT) / "scripts" / "validate_candidates_v13.py"
    spec = importlib.util.spec_from_file_location("_v13_validate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_v13_validate"] = mod
    spec.loader.exec_module(mod)
    return mod


def latex_escape(s: str) -> str:
    t = str(s)
    t = t.replace("\\", "\\textbackslash{}")
    t = t.replace("{", "\\{").replace("}", "\\}")
    t = t.replace("_", "\\_")
    t = t.replace("%", "\\%")
    t = t.replace("#", "\\#")
    t = t.replace("&", "\\&")
    t = t.replace("$", "\\$")
    return t


def _find_pdflatex() -> Optional[str]:
    w = shutil.which("pdflatex")
    if w:
        return w
    for p in ("/Library/TeX/texbin/pdflatex", "/usr/local/texlive/2026/bin/universal-darwin/pdflatex"):
        if Path(p).is_file():
            return p
    return None


def try_pdflatex(tex_path: Path, out_dir: Path) -> bool:
    exe = _find_pdflatex()
    if not exe:
        return False
    try:
        r = subprocess.run(
            [exe, "-interaction=nonstopmode", f"-output-directory={out_dir.resolve()}", tex_path.name],
            cwd=str(out_dir.resolve()),
            capture_output=True,
            text=True,
            timeout=180,
        )
        return r.returncode == 0 and (out_dir / "controls_report.pdf").is_file()
    except (OSError, subprocess.TimeoutExpired):
        return False


def spectral_only_metrics(v: Any, eig: np.ndarray, zeros: np.ndarray) -> Dict[str, Any]:
    """Spectral / spacing metrics given sorted eig and target zeros (same logic as validate_one)."""
    out: Dict[str, Any] = {
        "spectral_raw_mse": float("nan"),
        "spectral_log_mse": float("nan"),
        "spacing_mse_normalized": float("nan"),
        "ks_wigner": float("nan"),
    }
    z = np.asarray(zeros, dtype=_DTF).reshape(-1)
    z = z[np.isfinite(z)]
    eig = np.asarray(eig, dtype=_DTF).reshape(-1)
    eig = np.sort(eig[np.isfinite(eig)])
    k = int(min(int(eig.size), int(z.size)))
    if k < 1:
        return out
    e_k = np.sort(eig[:k])
    z_k = np.sort(z[:k])
    raw_mse = float(np.mean((e_k - z_k) ** 2))
    out["spectral_raw_mse"] = raw_mse
    out["spectral_log_mse"] = float(np.log1p(max(0.0, raw_mse)))
    if k >= 2:
        out["spacing_mse_normalized"] = float(v.spacing_mse_normalized(eig[:k], z[:k]))
        nu = v.normalized_gaps(np.sort(eig[:k]))
        if nu.size >= 2:
            out["ks_wigner"] = float(v.ks_against_wigner_gue(nu))
    return out


def safe_validate(
    v: Any,
    cand: Dict[str, Any],
    *,
    zeros: np.ndarray,
    dim: int,
    eps: float,
    seed: int,
    geo_weight: float,
    geo_sigma: float,
    potential_weight: float,
) -> Tuple[Dict[str, Any], Optional[str], Optional[np.ndarray]]:
    try:
        row, Hm = v.validate_one(
            cand,
            zeros=zeros,
            dim=int(dim),
            eps=float(eps),
            seed=int(seed),
            geo_weight=float(geo_weight),
            geo_sigma=float(geo_sigma),
            potential_weight=float(potential_weight),
        )
        return row, None, Hm
    except Exception as ex:
        return {"id": cand.get("id"), "word": cand.get("word"), "eig_error": repr(ex)}, repr(ex), None


def random_word(length: int, rng: np.random.Generator) -> List[int]:
    choices = [i for i in range(-6, 7) if i != 0]
    return [int(rng.choice(choices)) for _ in range(int(length))]


def load_candidate_bundle(
    candidate_path: Path,
    pareto_path: Optional[Path],
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    """Return (base candidates, spectral_id, geometry_id) for reporting."""
    with open(candidate_path, "r", encoding="utf-8") as f:
        cj = json.load(f)
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(cid: str, word: List[int]) -> None:
        if cid in seen:
            return
        seen.add(cid)
        out.append({"id": cid, "word": list(word)})

    sc = cj.get("spectral_candidate") if isinstance(cj.get("spectral_candidate"), dict) else {}
    gc = cj.get("geometric_candidate") if isinstance(cj.get("geometric_candidate"), dict) else {}
    if sc.get("id") and isinstance(sc.get("word"), list):
        add(str(sc["id"]), [int(x) for x in sc["word"]])
    if gc.get("id") and isinstance(gc.get("word"), list):
        add(str(gc["id"]), [int(x) for x in gc["word"]])

    want_extra = {"seed_14", "seed_20"}
    by_pareto: Dict[str, List[int]] = {}
    if pareto_path and pareto_path.is_file():
        with open(pareto_path, "r", encoding="utf-8") as f:
            pj = json.load(f)
        for row in pj.get("candidates") or []:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("id", ""))
            w = row.get("word_used") or row.get("word")
            if rid in want_extra and isinstance(w, list):
                by_pareto[rid] = [int(x) for x in w]

    for rid in sorted(want_extra):
        if rid in by_pareto and rid not in seen:
            add(rid, by_pareto[rid])

    sid = str(sc["id"]) if sc.get("id") else None
    gid = str(gc["id"]) if gc.get("id") else None
    return out, sid, gid


def numeric_stats(vals: List[float]) -> Dict[str, float]:
    xs = [float(x) for x in vals if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "median": float("nan"), "max": float("nan")}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "median": float(np.median(a)),
        "max": float(np.max(a)),
    }


def percentile_lower_better(control_vals: List[float], candidate_val: float) -> float:
    """Fraction of control draws with value >= candidate (higher = candidate beats more of the null)."""
    xs = [float(x) for x in control_vals if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if not xs or not math.isfinite(float(candidate_val)):
        return float("nan")
    return float(np.mean(np.asarray(xs, dtype=np.float64) >= float(candidate_val)))


def percentile_lower_better_geom(control_vals: List[float], candidate_val: float) -> float:
    """For nijenhuis/comm lower is better: fraction of controls with value >= candidate."""
    return percentile_lower_better(control_vals, candidate_val)


def row_to_flat(r: Dict[str, Any], *, control_group: str, sample_index: int, base_id: str) -> Dict[str, Any]:
    d = {
        "control_group": control_group,
        "sample_index": int(sample_index),
        "base_id": str(base_id),
        "id": str(r.get("id", "")),
        "eig_error": r.get("eig_error"),
    }
    for k in METRIC_KEYS:
        d[k] = r.get(k)
    if isinstance(d.get("operator_hash"), str):
        pass
    else:
        d["operator_hash"] = None
    return d


def gue_sample_metrics(
    v: Any,
    *,
    dim: int,
    zeros: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    G = rng.standard_normal((int(dim), int(dim)))
    H = (G + G.T) / math.sqrt(2.0)
    H = np.asarray(H, dtype=_DTF)
    eig = np.linalg.eigvalsh(H).astype(_DTF)
    eig = np.sort(eig[np.isfinite(eig)])
    sm = spectral_only_metrics(v, eig, zeros)
    hsh = hashlib.sha256(np.ascontiguousarray(H.astype(np.float64)).tobytes()).hexdigest()
    out = {
        "id": "gue",
        "word": None,
        "operator_builder": "gue_symmetric_gaussian",
        "operator_hash": hsh,
        "eig_min": float(np.min(eig)) if eig.size else float("nan"),
        "eig_max": float(np.max(eig)) if eig.size else float("nan"),
        "n_eig": int(eig.size),
        "n_zeros": int(np.asarray(zeros).size),
        "k_align": int(min(eig.size, np.asarray(zeros).size)),
        "ramsey_score": float("nan"),
        "nijenhuis_defect": float("nan"),
        "comm_norm_proxy": float("nan"),
        "eig_error": None,
    }
    out.update(sm)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="V13I control experiments for V13H candidates.")
    ap.add_argument("--candidate_json", type=str, default="runs/v13_candidate_theorem/candidate_operators.json")
    ap.add_argument("--pareto_json", type=str, default="runs/v13_top5_pareto/pareto_results.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13_controls")
    ap.add_argument("--zeros", type=int, default=128)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n_random", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--geo_weight", type=float, default=10.0)
    ap.add_argument("--geo_sigma", type=float, default=0.6)
    ap.add_argument("--potential_weight", type=float, default=0.25)
    args = ap.parse_args()

    v = _load_v13_validate()
    if v.build_word_sensitive_operator is None:
        raise SystemExit("word_sensitive operator builder required")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path = Path(args.candidate_json)
    if not cand_path.is_absolute():
        cand_path = Path(ROOT) / cand_path
    pareto_path = Path(args.pareto_json)
    if not pareto_path.is_absolute():
        pareto_path = Path(ROOT) / pareto_path

    zeros = v._load_zeros(int(args.zeros))
    if zeros.size < int(args.dim):
        raise ValueError(f"need zeros >= dim, got {zeros.size} < {args.dim}")

    base_list, spectral_id, geometry_id = load_candidate_bundle(
        cand_path, pareto_path if pareto_path.is_file() else None
    )
    if not base_list:
        raise SystemExit("no candidates loaded from candidate_json / pareto_json")

    lengths = sorted({len(c["word"]) for c in base_list})
    rng = np.random.default_rng(int(args.seed))

    rows: List[Dict[str, Any]] = []
    op_seed = int(args.seed)

    def log_err(msg: str) -> None:
        print(f"[v13-controls] {msg}", flush=True)

    eig_cache: Dict[Tuple[int, ...], np.ndarray] = {}
    geom_cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    # --- A: candidate ---
    for c in base_list:
        r, err, Hm = safe_validate(
            v,
            c,
            zeros=zeros,
            dim=int(args.dim),
            eps=float(args.eps),
            seed=op_seed,
            geo_weight=float(args.geo_weight),
            geo_sigma=float(args.geo_sigma),
            potential_weight=float(args.potential_weight),
        )
        if err:
            log_err(f"candidate {c.get('id')}: {err}")
        rows.append(row_to_flat(r, control_group="candidate", sample_index=0, base_id=str(c["id"])))
        key = tuple(int(x) for x in c["word"])
        if Hm is not None and not r.get("eig_error"):
            try:
                Ht = v._symmetrize_torch(Hm)
                eig_c, eerr = v._eigvalsh_safe(Ht)
                if not eerr and eig_c.size > 0:
                    eig_cache[key] = np.sort(eig_c)
                    geom_cache[key] = {
                        "ramsey_score": r.get("ramsey_score"),
                        "nijenhuis_defect": r.get("nijenhuis_defect"),
                        "comm_norm_proxy": r.get("comm_norm_proxy"),
                        "operator_hash": r.get("operator_hash"),
                        "eig_min": r.get("eig_min"),
                        "eig_max": r.get("eig_max"),
                        "n_eig": r.get("n_eig"),
                        "n_zeros": r.get("n_zeros"),
                    }
            except Exception as ex:
                log_err(f"eig_cache {c['id']}: {ex!r}")

    # --- B: random_words ---
    for i in range(int(args.n_random)):
        L = int(rng.choice(lengths))
        w = random_word(L, rng)
        cand = {"id": f"random_{i}", "word": w}
        r, err, _ = safe_validate(
            v,
            cand,
            zeros=zeros,
            dim=int(args.dim),
            eps=float(args.eps),
            seed=op_seed,
            geo_weight=float(args.geo_weight),
            geo_sigma=float(args.geo_sigma),
            potential_weight=float(args.potential_weight),
        )
        if err:
            log_err(f"random_{i}: {err}")
        rows.append(row_to_flat(r, control_group="random_words", sample_index=i, base_id="pool"))

    # --- C: shuffled_zeros (cached eig from candidate pass) ---
    for i in range(int(args.n_random)):
        c = base_list[int(rng.integers(0, len(base_list)))]
        key = tuple(int(x) for x in c["word"])
        if key not in eig_cache:
            continue
        z2 = np.asarray(zeros, dtype=_DTF).copy()
        rng.shuffle(z2)
        sm = spectral_only_metrics(v, eig_cache[key], z2)
        g = geom_cache[key]
        r = {
            "id": f"shuffle_{i}_{c['id']}",
            "word": list(key),
            "operator_builder": "word_sensitive",
            **sm,
            **{k: g.get(k) for k in ("ramsey_score", "nijenhuis_defect", "comm_norm_proxy", "operator_hash", "eig_min", "eig_max", "n_eig", "n_zeros")},
            "k_align": int(min(len(eig_cache[key]), z2.size)),
            "eig_error": None,
        }
        rows.append(row_to_flat(r, control_group="shuffled_zeros", sample_index=i, base_id=str(c["id"])))

    # --- D: sign_flips ---
    for c in base_list:
        w2 = [-int(x) for x in c["word"]]
        cand = {"id": f"signflip_{c['id']}", "word": w2}
        r, err, _ = safe_validate(
            v,
            cand,
            zeros=zeros,
            dim=int(args.dim),
            eps=float(args.eps),
            seed=op_seed,
            geo_weight=float(args.geo_weight),
            geo_sigma=float(args.geo_sigma),
            potential_weight=float(args.potential_weight),
        )
        if err:
            log_err(f"signflip {c['id']}: {err}")
        rows.append(row_to_flat(r, control_group="sign_flips", sample_index=0, base_id=str(c["id"])))

    # --- E: permuted_words ---
    for i in range(int(args.n_random)):
        c = base_list[int(rng.integers(0, len(base_list)))]
        w = list(c["word"])
        if len(w) <= 1:
            continue
        perm = rng.permutation(len(w))
        w2 = [w[int(j)] for j in perm]
        cand = {"id": f"perm_{i}_{c['id']}", "word": w2}
        r, err, _ = safe_validate(
            v,
            cand,
            zeros=zeros,
            dim=int(args.dim),
            eps=float(args.eps),
            seed=op_seed,
            geo_weight=float(args.geo_weight),
            geo_sigma=float(args.geo_sigma),
            potential_weight=float(args.potential_weight),
        )
        if err:
            log_err(f"perm_{i}: {err}")
        rows.append(row_to_flat(r, control_group="permuted_words", sample_index=i, base_id=str(c["id"])))

    # --- F: GUE baseline ---
    for i in range(int(args.n_random)):
        try:
            r = gue_sample_metrics(v, dim=int(args.dim), zeros=zeros, rng=rng)
            r["id"] = f"gue_{i}"
            rows.append(row_to_flat(r, control_group="gue_spacing_baseline", sample_index=i, base_id="gue"))
        except Exception as ex:
            log_err(f"gue_{i}: {ex!r}")

    # --- aggregate ---
    groups = sorted({str(r["control_group"]) for r in rows})
    numeric_keys = [k for k in METRIC_KEYS if k != "operator_hash"]
    by_group: Dict[str, Any] = {g: {} for g in groups}
    for g in groups:
        grp_rows = [r for r in rows if r["control_group"] == g]
        for mk in numeric_keys:
            vals = []
            for r in grp_rows:
                x = r.get(mk)
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    vals.append(float(x))
            by_group[g][mk] = numeric_stats(vals)

    cand_rows = [r for r in rows if r["control_group"] == "candidate"]
    rand_rows = [r for r in rows if r["control_group"] == "random_words"]
    percentiles: Dict[str, Any] = {}
    for cr in cand_rows:
        cid = str(cr["id"])
        percentiles[cid] = {}
        for mk in ("spectral_log_mse", "ks_wigner", "comm_norm_proxy", "nijenhuis_defect", "spacing_mse_normalized"):
            rv = cr.get(mk)
            if not isinstance(rv, (int, float)) or not math.isfinite(float(rv)):
                percentiles[cid][mk] = None
                continue
            ctrl = [float(x) for x in (r.get(mk) for r in rand_rows) if isinstance(x, (int, float)) and math.isfinite(float(x))]
            percentiles[cid][mk] = percentile_lower_better(ctrl, float(rv))

    interpretation = {
        "warning": "Computational control evidence only; not a proof of the Riemann Hypothesis.",
        "spectral_vs_random": [],
        "geometry_vs_random": [],
    }
    sid_s = spectral_id or ""
    gid_s = geometry_id or ""
    sc_row = next((r for r in cand_rows if str(r["id"]) == sid_s), None)
    gc_row = next((r for r in cand_rows if str(r["id"]) == gid_s), None)
    if sc_row and rand_rows:
        msl = float(sc_row["spectral_log_mse"]) if math.isfinite(float(sc_row.get("spectral_log_mse", float("nan")))) else float("nan")
        rvals = [float(r["spectral_log_mse"]) for r in rand_rows if math.isfinite(float(r.get("spectral_log_mse", float("nan"))))]
        med = float(np.median(np.asarray(rvals))) if rvals else float("nan")
        interpretation["spectral_vs_random"].append(
            f"{sid_s} spectral_log_mse={msl:.6g}; random_words median={med:.6g}; "
            f"fraction of random with worse (>=) mse: {percentiles.get(sid_s, {}).get('spectral_log_mse', 'n/a')}."
        )
        interpretation["spectral_vs_random"].append(
            "Lower spectral_log_mse is better; if the fraction is high, the spectral candidate beats most random nulls."
        )
    if sc_row and rand_rows:
        ks_c = float(sc_row["ks_wigner"]) if math.isfinite(float(sc_row.get("ks_wigner", float("nan")))) else float("nan")
        ks_r = [float(r["ks_wigner"]) for r in rand_rows if math.isfinite(float(r.get("ks_wigner", float("nan"))))]
        medk = float(np.median(np.asarray(ks_r))) if ks_r else float("nan")
        interpretation["spectral_vs_random"].append(
            f"{sid_s} ks_wigner={ks_c:.6g}; random median={medk:.6g}; beat-random fraction (>=): {percentiles.get(sid_s, {}).get('ks_wigner', 'n/a')}."
        )
    if gc_row and rand_rows:
        for mk, label in (("comm_norm_proxy", "comm_norm_proxy"), ("nijenhuis_defect", "nijenhuis_defect")):
            cv = float(gc_row.get(mk, float("nan")))
            ctrl = [float(r[mk]) for r in rand_rows if math.isfinite(float(r.get(mk, float("nan"))))]
            med = float(np.median(np.asarray(ctrl))) if ctrl else float("nan")
            frac = percentile_lower_better(ctrl, cv) if math.isfinite(cv) else float("nan")
            interpretation["geometry_vs_random"].append(
                f"{gid_s} {label}={cv:.6g} vs random median={med:.6g}; fraction random >= candidate: {frac:.4f} (lower is better for geometry candidate)."
            )

    payload = {
        "meta": {
            "zeros_n": int(zeros.size),
            "dim": int(args.dim),
            "n_random": int(args.n_random),
            "seed": int(args.seed),
            "operator_builder": "word_sensitive",
            "spectral_candidate_id": spectral_id,
            "geometric_candidate_id": geometry_id,
            "base_candidates": [{"id": c["id"], "word": c["word"]} for c in base_list],
        },
        "by_group_stats": by_group,
        "candidate_percentiles_vs_random_words": percentiles,
        "interpretation": interpretation,
        "rows": rows,
    }

    with open(out_dir / "controls_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)

    # CSV summary: long format
    with open(out_dir / "controls_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["control_group", "metric", "mean", "std", "min", "median", "max"])
        for g in groups:
            for mk in numeric_keys:
                st = by_group[g].get(mk) or {}
                w.writerow([g, mk, st.get("mean"), st.get("std"), st.get("min"), st.get("median"), st.get("max")])

    # Markdown report
    md_lines = [
        "# V13I Control Study: V13H Candidates vs Nulls\n\n",
        "## Status\n\n",
        f"> {interpretation['warning']}\n\n",
        "## Design\n\n",
        "- **candidate**: validated spectral/geometric (+ pareto extras) words, standard zeros.\n",
        "- **random_words**: `n_random` words with lengths drawn from candidate length set; generators uniform in {-6,...,6} excluding 0.\n",
        "- **shuffled_zeros**: fixed word-sensitive `H_w`; target ordinates permuted before spectral/spacing metrics.\n",
        "- **sign_flips**: `a_i -> -a_i` on each base candidate word.\n",
        "- **permuted_words**: random permutations of base candidate letters.\n",
        "- **gue_spacing_baseline**: i.i.d. symmetric Gaussian matrices, spacing/KS vs true zeros.\n\n",
        "## Candidate percentiles vs random (spectral/KS)\n\n",
    ]
    for k, vdict in percentiles.items():
        md_lines.append(f"### `{k}`\n\n")
        for mk, pv in (vdict or {}).items():
            md_lines.append(f"- `{mk}`: fraction of random_words with **worse or equal** value (lower is better): **{pv}**\n")
        md_lines.append("\n")
    md_lines.append("## Automated interpretation\n\n")
    for line in interpretation.get("spectral_vs_random", []):
        md_lines.append(f"- {line}\n")
    for line in interpretation.get("geometry_vs_random", []):
        md_lines.append(f"- {line}\n")
    md_lines.append("\n## Group statistics (spectral_log_mse)\n\n")
    md_lines.append("| control_group | mean | median | std |\n|---|---|---|---|\n")
    mk = "spectral_log_mse"
    for g in groups:
        st = by_group.get(g, {}).get(mk) or {}
        md_lines.append(
            f"| `{g}` | {st.get('mean', '')} | {st.get('median', '')} | {st.get('std', '')} |\n"
        )
    md_lines.append("\nFull tables: `controls_summary.csv` and `controls_results.json` (`by_group_stats`).\n")

    (out_dir / "controls_report.md").write_text("".join(md_lines), encoding="utf-8")

    # LaTeX
    tex_parts: List[str] = [
        "\\documentclass[11pt]{article}\n",
        "\\usepackage{amsmath,amssymb}\n",
        "\\usepackage[T1]{fontenc}\n",
        "\\usepackage[margin=1in]{geometry}\n",
        "\\title{V13I Control Study}\n\\date{}\n\\begin{document}\n\\maketitle\n",
        "\\section*{Warning}\n",
        latex_escape(interpretation["warning"]) + "\n\n",
        "\\section{Spectral log MSE by control group}\n",
        "\\begin{tabular}{|l|r|r|r|}\\hline\n",
        "\\textbf{group} & \\textbf{mean} & \\textbf{median} & \\textbf{std} \\\\\n\\hline\n",
    ]
    mk0 = "spectral_log_mse"
    for g in groups:
        st = by_group.get(g, {}).get(mk0) or {}
        tex_parts.append(
            latex_escape(g)
            + " & "
            + latex_escape(str(st.get("mean", "")))
            + " & "
            + latex_escape(str(st.get("median", "")))
            + " & "
            + latex_escape(str(st.get("std", "")))
            + " \\\\\n\\hline\n"
        )
    tex_parts.append("\\end{tabular}\n\n")
    tex_parts.append(
        "\\section{Files}\nSee \\texttt{controls\\_summary.csv} and \\texttt{controls\\_results.json}.\n\n"
        "\\end{document}\n"
    )
    (out_dir / "controls_report.tex").write_text("".join(tex_parts), encoding="utf-8")

    if try_pdflatex(out_dir / "controls_report.tex", out_dir):
        print(f"Wrote {out_dir / 'controls_report.pdf'}")
    else:
        print("PDF skipped (pdflatex missing or failed).")

    print(f"Wrote {out_dir / 'controls_results.json'}")
    print(f"Wrote {out_dir / 'controls_summary.csv'}")
    print(f"Wrote {out_dir / 'controls_report.md'}")
    print(f"Wrote {out_dir / 'controls_report.tex'}")


if __name__ == "__main__":
    main()
