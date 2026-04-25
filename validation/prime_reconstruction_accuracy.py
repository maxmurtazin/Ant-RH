from __future__ import annotations

"""Prime reconstruction diagnostics from DTES spectral geometry.

This module is a numerical reconstruction diagnostic only. It is not a
primality test replacement and does not claim or prove RH.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def _as_finite_1d(values: Any) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _sanitize_gammas(values: Any, max_zeros: int | None = None) -> np.ndarray:
    gammas = np.abs(_as_finite_1d(values))
    gammas = np.sort(gammas[gammas > 0.0])
    if max_zeros is not None and max_zeros > 0:
        gammas = gammas[: int(max_zeros)]
    return gammas


def chebyshev_psi_from_zeros(
    x_grid: Sequence[float],
    zeta_zeros: Sequence[float],
    max_zeros: int | None = None,
) -> np.ndarray:
    """Approximate psi(x) using RH-form zeros rho = 1/2 + i gamma.

    Uses the real-valued symmetrized explicit-formula-inspired form:
    psi(x) ~= x - 2 * Re(sum_gamma x^(1/2+i gamma) / (1/2+i gamma)).
    """

    x = _as_finite_1d(x_grid)
    if x.size == 0:
        return np.array([], dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("x_grid must contain positive values")

    gammas = _sanitize_gammas(zeta_zeros, max_zeros=max_zeros)
    psi = x.astype(float).copy()
    if gammas.size == 0:
        return psi

    log_x = np.log(x)
    sqrt_x = np.sqrt(x)
    rho = 0.5 + 1j * gammas
    total = np.zeros(x.shape, dtype=np.complex128)

    # Chunking keeps memory bounded for larger grids or zero lists.
    chunk_size = 256
    for start in range(0, gammas.size, chunk_size):
        rho_chunk = rho[start : start + chunk_size]
        terms = sqrt_x[:, None] * np.exp(1j * log_x[:, None] * gammas[start : start + chunk_size])
        total += np.sum(terms / rho_chunk[None, :], axis=1)

    psi = psi - 2.0 * np.real(total)
    return np.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)


def von_mangoldt_from_psi(
    x_grid: Sequence[float],
    psi_values: Sequence[float],
) -> np.ndarray:
    """Approximate Lambda(n) by the discrete difference psi(n) - psi(n-1)."""

    x = _as_finite_1d(x_grid)
    psi = _as_finite_1d(psi_values)
    if x.size != psi.size:
        raise ValueError("x_grid and psi_values must have the same length")
    if psi.size == 0:
        return np.array([], dtype=float)

    order = np.argsort(x)
    lambda_sorted = np.diff(psi[order], prepend=0.0)
    lambda_values = np.empty_like(lambda_sorted)
    lambda_values[order] = lambda_sorted
    return np.nan_to_num(lambda_values, nan=0.0, posinf=0.0, neginf=0.0)


def reconstruct_prime_candidates(
    x_grid: Sequence[float],
    lambda_values: Sequence[float],
    threshold_mode: str = "adaptive",
) -> List[int]:
    """Return integers whose reconstructed Lambda(n) is close to log(n)."""

    x = _as_finite_1d(x_grid)
    lambdas = _as_finite_1d(lambda_values)
    if x.size != lambdas.size:
        raise ValueError("x_grid and lambda_values must have the same length")
    if x.size == 0:
        return []

    n = np.rint(x).astype(int)
    valid = (n >= 2) & np.isclose(x, n, atol=1e-9)
    if not np.any(valid):
        return []

    n_valid = n[valid]
    lambda_valid = lambdas[valid]
    log_n = np.log(n_valid.astype(float))
    scores = np.abs(lambda_valid - log_n) / np.maximum(log_n, 1e-12)

    if threshold_mode == "adaptive":
        x_max = int(np.max(n_valid))
        expected = _prime_count_estimate(x_max)
        k = int(np.clip(expected, 1, n_valid.size))
        chosen = np.argsort(scores)[:k]
        candidates = n_valid[chosen]
    elif threshold_mode == "relative":
        candidates = n_valid[scores <= 0.35]
    elif threshold_mode == "absolute":
        candidates = n_valid[np.abs(lambda_valid - log_n) <= 0.5]
    else:
        raise ValueError("threshold_mode must be one of: adaptive, relative, absolute")

    return sorted(int(v) for v in np.unique(candidates))


def exact_primes_up_to(n: int) -> List[int]:
    """Generate exact primes with a sieve for evaluation only."""

    limit = int(n)
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    root = int(math.isqrt(limit))
    for p in range(2, root + 1):
        if sieve[p]:
            start = p * p
            sieve[start : limit + 1 : p] = b"\x00" * (((limit - start) // p) + 1)
    return [i for i in range(2, limit + 1) if sieve[i]]


def prime_reconstruction_metrics(
    predicted_primes: Iterable[int],
    true_primes: Iterable[int],
) -> Dict[str, float | int]:
    predicted = sorted({int(p) for p in predicted_primes if int(p) >= 2})
    true = sorted({int(p) for p in true_primes if int(p) >= 2})
    pred_set = set(predicted)
    true_set = set(true)

    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    n_max = max(predicted[-1] if predicted else 1, true[-1] if true else 1)
    true_curve = _prime_count_curve(true, n_max)
    pred_curve = _prime_count_curve(predicted, n_max)
    diff = pred_curve - true_curve
    abs_diff = np.abs(diff)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_absolute_prime_count_error": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "max_prime_count_error": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "pi_curve_mae": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "pi_curve_rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "n_true_primes": int(len(true)),
        "n_predicted_primes": int(len(predicted)),
    }


def _prime_count_estimate(x_max: int) -> int:
    if x_max < 2:
        return 0
    if x_max < 17:
        return max(1, int(round(x_max / max(math.log(max(x_max, 3)), 1.0))))
    log_x = math.log(float(x_max))
    estimate = x_max / log_x + x_max / (log_x * log_x)
    return max(1, int(round(estimate)))


def _prime_count_curve(primes: Sequence[int], n_max: int) -> np.ndarray:
    curve = np.zeros(int(n_max) + 1, dtype=float)
    for p in primes:
        if 0 <= int(p) <= n_max:
            curve[int(p) :] += 1.0
    return curve[1:]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("spectral report must contain a JSON object")
    return data


def _first_present(data: Dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = data.get(name)
        if value is not None:
            return value
    return None


def _resolve_report_path(report_path: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = report_path.parent / path
    if candidate.exists():
        return candidate
    sibling = report_path.parent / path.name
    if sibling.exists():
        return sibling
    return path


def load_vector_from_csv(path) -> np.ndarray:
    csv_path = Path(path)
    rows: List[float] = []
    preferred_names = ("eigval", "eigenvalue", "lambda", "value")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        if has_header:
            reader = csv.DictReader(f)
            fieldnames = [name.strip() for name in (reader.fieldnames or [])]
            selected = next((name for name in fieldnames if name.lower() in preferred_names), None)
            if selected is None:
                selected = fieldnames[-1] if fieldnames else None
            if selected is None:
                return np.array([], dtype=float)
            for row in reader:
                try:
                    rows.append(float(row[selected]))
                except (KeyError, TypeError, ValueError):
                    continue
        else:
            reader = csv.reader(f)
            for row in reader:
                for cell in reversed(row):
                    try:
                        rows.append(float(cell))
                        break
                    except ValueError:
                        continue

    return _as_finite_1d(rows)


def _scale_to_zero_range(eigvals: np.ndarray, zeta_zeros: np.ndarray) -> np.ndarray:
    k = min(len(eigvals), len(zeta_zeros))
    e = np.sort(_as_finite_1d(eigvals))[:k]
    z = np.sort(_as_finite_1d(zeta_zeros))[:k]
    if k == 0:
        return np.array([], dtype=float)
    e_norm = (e - e.mean()) / (e.std() + 1e-12)
    reconstructed_zeros = e_norm * (z.std() + 1e-12) + z.mean()
    return _sanitize_gammas(reconstructed_zeros)


def load_prime_reconstruction_inputs(report_path) -> Dict[str, Any]:
    path = Path(report_path)
    report = _load_json(path)
    source_type = "operator_learning" if report.get("learned_eigenvalues") else "spectral_report"

    zeta_zeros = _sanitize_gammas(
        _first_present(report, ("zeta_zeros", "true_zeros", "known_zeros"))
    )
    eigvals = _sanitize_gammas(_first_present(report, ("eigvals", "dtes_eigvals", "eigenvalues")))
    learned_eigvals = np.array([], dtype=float)
    input_json_path = None

    learned_path_value = report.get("learned_eigenvalues")
    if eigvals.size == 0 and learned_path_value:
        learned_path = _resolve_report_path(path, learned_path_value)
        learned_eigvals = _sanitize_gammas(load_vector_from_csv(learned_path))
        eigvals = learned_eigvals
        print(f"[OK] loaded learned eigenvalues from {learned_path}")

    input_value = report.get("input")
    if zeta_zeros.size == 0 and input_value:
        input_json_path = _resolve_report_path(path, input_value)
        input_data = _load_json(input_json_path)
        zeta_zeros = _sanitize_gammas(
            _first_present(input_data, ("zeta_zeros", "true_zeros", "known_zeros"))
        )
        if zeta_zeros.size:
            print(f"[OK] loaded zeta_zeros from {input_json_path}")

    if zeta_zeros.size == 0:
        raise ValueError("zeta_zeros not found in report or input JSON")
    if zeta_zeros.size < 5:
        raise ValueError(f"at least 5 zeta_zeros are required, got {zeta_zeros.size}")
    if eigvals.size < 5:
        raise ValueError(f"at least 5 eigenvalues are required, got {eigvals.size}")

    learned_reconstructed = np.array([], dtype=float)
    if source_type == "operator_learning":
        learned_reconstructed = _scale_to_zero_range(eigvals, zeta_zeros)
        if learned_reconstructed.size < 5:
            raise ValueError(
                f"at least 5 normalized learned eigenvalues are required, got {learned_reconstructed.size}"
            )

    return {
        "source_type": source_type,
        "report": report,
        "report_path": path,
        "input_json_path": input_json_path,
        "zeta_zeros": np.sort(zeta_zeros),
        "eigvals": np.sort(eigvals),
        "learned_operator_zeros": learned_reconstructed,
    }


def _extract_zero_sources(report: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    zeta_zeros = _first_present(report, ("zeta_zeros", "true_zeros", "known_zeros"))
    eigvals = _first_present(report, ("eigvals", "dtes_eigvals", "eigenvalues"))
    return _sanitize_gammas(zeta_zeros), _sanitize_gammas(eigvals)


def _run_reconstruction(
    label: str,
    x_grid: np.ndarray,
    zeros: np.ndarray,
    true_primes: Sequence[int],
    threshold_mode: str,
    max_zeros: int | None,
) -> Dict[str, Any]:
    psi = chebyshev_psi_from_zeros(x_grid, zeros, max_zeros=max_zeros)
    lambdas = von_mangoldt_from_psi(x_grid, psi)
    candidates = reconstruct_prime_candidates(x_grid, lambdas, threshold_mode=threshold_mode)
    metrics = prime_reconstruction_metrics(candidates, true_primes)
    return {
        "label": label,
        "n_zeros_used": int(_sanitize_gammas(zeros, max_zeros=max_zeros).size),
        "metrics": metrics,
        "prime_candidates": candidates,
        "lambda_values": [float(v) for v in lambdas],
        "psi_values": [float(v) for v in psi],
    }


def _maybe_write_plots(
    out_path: Path,
    x_grid: np.ndarray,
    true_primes: Sequence[int],
    baseline: Dict[str, Any],
    dtes: Dict[str, Any],
) -> None:
    plot_cache = out_path.parent / ".matplotlib"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
    except BaseException:
        return

    out_dir = out_path.parent
    x_max = int(np.max(x_grid)) if x_grid.size else 0
    true_curve = _prime_count_curve(true_primes, x_max)
    base_curve = _prime_count_curve(baseline["prime_candidates"], x_max)
    dtes_curve = _prime_count_curve(dtes["prime_candidates"], x_max)
    dtes_label = dtes.get("label", "dtes_spectrum_reconstruction")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_grid, true_curve, label="true pi(x)")
    ax.plot(x_grid, base_curve, label="true-zero reconstruction")
    ax.plot(x_grid, dtes_curve, label=dtes_label.replace("_", " "))
    ax.set_title("Prime Count Curve: True vs Reconstructed")
    ax.set_xlabel("n")
    ax.set_ylabel("pi(n)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "prime_count_curve_true_vs_reconstructed.png", dpi=180)
    plt.close(fig)

    log_n = np.log(np.maximum(x_grid, 2.0))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_grid, log_n, label="log(n)", linewidth=1.0)
    ax.plot(x_grid, baseline["lambda_values"], label="true-zero Lambda(n)", alpha=0.8)
    ax.plot(x_grid, dtes["lambda_values"], label=f"{dtes_label} Lambda(n)", alpha=0.8)
    ax.set_title("Von Mangoldt Reconstruction")
    ax.set_xlabel("n")
    ax.set_ylabel("Lambda(n)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "von_mangoldt_reconstruction.png", dpi=180)
    plt.close(fig)

    true_set = set(int(p) for p in true_primes)
    reconstruction_candidates = [int(p) for p in dtes["prime_candidates"]]
    hits = [p for p in reconstruction_candidates if p in true_set]
    misses = [p for p in reconstruction_candidates if p not in true_set]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(true_primes, np.ones(len(true_primes)), s=8, label="true primes", alpha=0.45)
    ax.scatter(hits, np.full(len(hits), 1.1), s=14, label="reconstruction hits")
    ax.scatter(misses, np.full(len(misses), 0.9), s=14, label="reconstruction misses")
    ax.set_title("Prime Candidate Hits")
    ax.set_xlabel("n")
    ax.set_yticks([])
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "prime_candidate_hits.png", dpi=180)
    plt.close(fig)


def build_prime_reconstruction_report(args: argparse.Namespace) -> Dict[str, Any]:
    x_max = int(args.x_max)
    if x_max < 2:
        raise ValueError("--x-max must be at least 2")

    inputs = load_prime_reconstruction_inputs(Path(args.spectral_report))
    zeta_zeros = inputs["zeta_zeros"]
    dtes_eigvals = inputs["eigvals"]
    x_grid = np.arange(1, x_max + 1, dtype=float)
    true_primes = exact_primes_up_to(x_max)

    baseline = _run_reconstruction(
        "true_zero_baseline",
        x_grid,
        zeta_zeros,
        true_primes,
        threshold_mode=args.threshold_mode,
        max_zeros=args.max_zeros,
    )
    source_type = inputs["source_type"]

    learned = None
    dtes = None
    if source_type == "operator_learning":
        learned = _run_reconstruction(
            "learned_operator_reconstruction",
            x_grid,
            inputs["learned_operator_zeros"],
            true_primes,
            threshold_mode=args.threshold_mode,
            max_zeros=args.max_zeros,
        )
    else:
        dtes = _run_reconstruction(
            "dtes_spectrum_reconstruction",
            x_grid,
            dtes_eigvals,
            true_primes,
            threshold_mode=args.threshold_mode,
            max_zeros=args.max_zeros,
        )

    result = {
        "x_max": x_max,
        "source_type": source_type,
        "note": (
            "Prime reconstruction from learned spectrum is a numerical diagnostic, "
            "not a primality test or proof of RH."
        ),
        "diagnostic_warning": (
            "Prime reconstruction accuracy is a numerical diagnostic only; "
            "it is not a primality test replacement and does not prove RH."
        ),
        "separation_notes": {
            "zero_reconstruction_accuracy": "quality of zeros used in the explicit-formula reconstruction",
            "spectral_alignment_accuracy": "alignment of DTES operator eigenvalues with zeta zeros",
            "prime_reconstruction_accuracy": "ability of reconstructed Lambda/pi curves to recover prime-counting structure",
        },
        "true_zero_baseline": baseline,
        "true_primes": true_primes,
    }
    if learned is not None:
        result["learned_operator_reconstruction"] = learned
        result["prime_candidates"] = learned["prime_candidates"]
    if dtes is not None:
        result["dtes_spectrum_reconstruction"] = dtes
        result["prime_candidates"] = dtes["prime_candidates"]
    return result


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether reconstructed DTES spectral geometry recovers "
            "prime-counting structure. Diagnostic only; not a primality test."
        )
    )
    parser.add_argument("--spectral-report", required=True, help="Input spectral_report.json")
    parser.add_argument("--x-max", type=int, default=1000)
    parser.add_argument("--out", required=True, help="Output prime reconstruction report JSON")
    parser.add_argument("--max-zeros", type=int, default=None)
    parser.add_argument(
        "--threshold-mode",
        default="adaptive",
        choices=("adaptive", "relative", "absolute"),
    )
    args = parser.parse_args()

    try:
        report = build_prime_reconstruction_report(args)
    except Exception as exc:
        print(f"[ERROR] prime reconstruction failed: {exc}", file=sys.stderr)
        raise SystemExit(2)

    out_path = Path(args.out)
    _write_json(out_path, report)
    reconstruction = report.get("learned_operator_reconstruction") or report.get(
        "dtes_spectrum_reconstruction"
    )
    _maybe_write_plots(
        out_path,
        np.arange(1, int(args.x_max) + 1, dtype=float),
        report["true_primes"],
        report["true_zero_baseline"],
        reconstruction,
    )
    print(f"[OK] saved prime reconstruction report -> {args.out}")
    if reconstruction is not None:
        print(
            f"[OK] baseline_f1={report['true_zero_baseline']['metrics']['f1']:.3f} "
            f"reconstruction_f1={reconstruction['metrics']['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
