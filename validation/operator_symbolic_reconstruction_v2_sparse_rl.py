from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import signal
import shutil
import subprocess
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm

    HAS_TQDM = True
except Exception:
    tqdm = None
    HAS_TQDM = False

try:
    import sympy

    HAS_SYMPY = True
except Exception:
    sympy = None
    HAS_SYMPY = False


EPS = 1e-12
MAX_DENSE_EIG_N = 1800
DEFAULT_FIRST_K = 50
TITLE = "Symbolic kernel candidate — not a proof of RH"


def warn_if_slow(stage_time: float, threshold: float = 10.0) -> None:
    """Warn when a pipeline stage takes longer than expected."""
    if stage_time > threshold:
        print(f"[V11.2][WARN] Stage slow: {stage_time:.2f}s")


class StageLogger:
    """Global stage logger for V11.2 timing diagnostics."""

    def __init__(self) -> None:
        self.global_start = time.time()
        self.stage_name = ""
        self.stage_start = self.global_start

    def reset(self) -> None:
        """Reset the global timer for a fresh CLI run."""
        self.global_start = time.time()
        self.stage_name = ""
        self.stage_start = self.global_start

    def start(self, name: str) -> None:
        """Mark the start of a named stage."""
        self.stage_name = name
        self.stage_start = time.time()
        print(f"\n[V11.2] ▶ START: {name}")

    def end(self) -> float:
        """Mark the end of the current stage and return stage duration."""
        dt = time.time() - self.stage_start
        total = time.time() - self.global_start
        warn_if_slow(dt)
        print(f"[V11.2] ✔ END: {self.stage_name} | stage_time={dt:.2f}s | total={total:.2f}s")
        return dt


stage_logger = StageLogger()


@dataclass(frozen=True)
class FeatureSpec:
    """Description of one real-valued kernel basis feature."""

    name: str
    evaluator: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    sympy_builder: Callable[[Any, Any, Any], Any]
    wolfram: str


@dataclass
class RegressionResult:
    """Sparse complex regression result in the unscaled feature basis."""

    coefficients: np.ndarray
    selected: np.ndarray
    mse: float
    score: float
    method: str


def _json_safe(value: Any) -> Any:
    """Convert NumPy and non-finite values into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (complex, np.complexfloating)):
        value = complex(value)
        return {
            "real": float(value.real) if math.isfinite(value.real) else None,
            "imag": float(value.imag) if math.isfinite(value.imag) else None,
        }
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the V11.2 pipeline."""
    parser = argparse.ArgumentParser(
        description="V11.2 symbolic operator reconstruction with sparse regression and RL feedback."
    )
    parser.add_argument("--operator", required=True, help="Path to learned complex operator .npy")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--fourier-k", type=int, default=5, help="Maximum k for sin(k Delta), cos(k Delta)")
    parser.add_argument("--alpha-sparsity", type=float, default=1e-4, help="L0 sparsity penalty")
    parser.add_argument("--lambda-spec", type=float, default=1.0, help="Spectral error reward weight")
    parser.add_argument("--lambda-complexity", type=float, default=0.01, help="Term-count reward weight")
    parser.add_argument("--lambda-herm", type=float, default=10.0, help="Hermitian error reward weight")
    parser.add_argument("--max-pairs", type=int, default=200_000, help="Maximum sampled matrix entries for regression")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pair subsampling")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Worker processes")
    parser.add_argument("--batch-size", type=int, default=50_000, help="Feature/kernel pair batch size")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bars or ETA logs")
    parser.add_argument("--no-multiprocessing", action="store_true", help="Disable multiprocessing acceleration")
    parser.add_argument("--sympy-mode", choices=["none", "fast", "safe", "full"], default="safe", help="Symbolic simplification mode")
    parser.add_argument("--max-symbolic-terms", type=int, default=12, help="Maximum terms allowed in displayed symbolic expression")
    parser.add_argument("--max-symbolic-ops", type=int, default=250, help="Maximum operation count before guarded simplification")
    parser.add_argument("--simplify-timeout", type=float, default=20.0, help="Per-operation SymPy simplification timeout in seconds")
    return parser.parse_args(argv)


def progress_iter(iterable: Iterable[Any], total: Optional[int] = None, desc: str = "progress", enabled: bool = True) -> Iterable[Any]:
    """Wrap an iterable with tqdm when available and requested."""
    if enabled and HAS_TQDM and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


class ETATracker:
    """Minimal progress logger used when tqdm is unavailable."""

    def __init__(self, total: int, name: str = "progress", enabled: bool = True) -> None:
        self.total = int(total)
        self.name = name
        self.enabled = bool(enabled)
        self.start = time.time()
        self.last = self.start

    def update(self, step: int) -> None:
        """Print a throttled progress/ETA line."""
        if not self.enabled or self.total <= 0:
            return
        now = time.time()
        if now - self.last < 1.0 and step < self.total:
            return
        elapsed = now - self.start
        rate = step / max(elapsed, 1e-8)
        eta = (self.total - step) / max(rate, 1e-8)
        print(
            f"[{self.name}] {step}/{self.total} "
            f"({100.0 * step / self.total:.1f}%) "
            f"elapsed={elapsed:.1f}s ETA={eta:.1f}s"
        )
        self.last = now


def make_batches(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    """Yield consecutive batches from a sequence."""
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_operator(path: Path) -> np.ndarray:
    """Load and validate a finite square complex operator matrix."""
    if not path.exists():
        raise FileNotFoundError(f"operator file does not exist: {path}")
    H = np.load(path, mmap_mode="r")
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"operator must be a square matrix, got shape {H.shape}")
    H_arr = np.asarray(H, dtype=np.complex128)
    if not np.all(np.isfinite(H_arr)):
        bad = int(np.size(H_arr) - np.count_nonzero(np.isfinite(H_arr)))
        raise FloatingPointError(f"operator contains {bad} NaN/Inf entries")
    return H_arr


def hermitian_projection(H: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return Hermitian projection and relative Frobenius Hermitian defect."""
    denom = float(np.linalg.norm(H))
    error = float(np.linalg.norm(H - H.conj().T) / max(denom, EPS))
    H_herm = 0.5 * (H + H.conj().T)
    return H_herm, error


def build_feature_specs(fourier_k: int, epsilon: float = 1e-6) -> List[FeatureSpec]:
    """Create the real-valued symbolic feature dictionary for K(t_i, t_j)."""
    if fourier_k < 0:
        raise ValueError("--fourier-k must be non-negative")

    specs: List[FeatureSpec] = [
        FeatureSpec("1", lambda ti, tj, d: np.ones_like(d), lambda ti, tj, d: 1, "1"),
        FeatureSpec("Delta", lambda ti, tj, d: d, lambda ti, tj, d: d, "Delta"),
        FeatureSpec("Delta^2", lambda ti, tj, d: d**2, lambda ti, tj, d: d**2, "Delta^2"),
        FeatureSpec(
            "log(1+Delta)",
            lambda ti, tj, d: np.log1p(d),
            lambda ti, tj, d: __import__("sympy").log(1 + d),
            "Log[1 + Delta]",
        ),
        FeatureSpec(
            "1/(1+Delta)",
            lambda ti, tj, d: 1.0 / (1.0 + d),
            lambda ti, tj, d: 1 / (1 + d),
            "1/(1 + Delta)",
        ),
        FeatureSpec(
            "exp(-Delta^2)",
            lambda ti, tj, d: np.exp(-(d**2)),
            lambda ti, tj, d: __import__("sympy").exp(-(d**2)),
            "Exp[-Delta^2]",
        ),
    ]

    for k in range(1, fourier_k + 1):
        specs.append(
            FeatureSpec(
                f"cos({k}*Delta)",
                lambda ti, tj, d, kk=k: np.cos(kk * d),
                lambda ti, tj, d, kk=k: __import__("sympy").cos(kk * d),
                f"Cos[{k} Delta]",
            )
        )
        specs.append(
            FeatureSpec(
                f"sin({k}*Delta)",
                lambda ti, tj, d, kk=k: np.sin(kk * d),
                lambda ti, tj, d, kk=k: __import__("sympy").sin(kk * d),
                f"Sin[{k} Delta]",
            )
        )

    specs.extend(
        [
            FeatureSpec(
                "1/(epsilon+Delta)",
                lambda ti, tj, d, eps=epsilon: 1.0 / (eps + d),
                lambda ti, tj, d, eps=epsilon: 1 / (eps + d),
                f"1/({epsilon:.17g} + Delta)",
            ),
            FeatureSpec(
                "1/(epsilon+Delta^2)",
                lambda ti, tj, d, eps=epsilon: 1.0 / (eps + d**2),
                lambda ti, tj, d, eps=epsilon: 1 / (eps + d**2),
                f"1/({epsilon:.17g} + Delta^2)",
            ),
            FeatureSpec(
                "cos(t_i*log(1+t_j))",
                lambda ti, tj, d: np.cos(ti * np.log1p(tj)),
                lambda ti, tj, d: __import__("sympy").cos(ti * __import__("sympy").log(1 + tj)),
                "Cos[ti Log[1 + tj]]",
            ),
            FeatureSpec(
                "cos(t_j*log(1+t_i))",
                lambda ti, tj, d: np.cos(tj * np.log1p(ti)),
                lambda ti, tj, d: __import__("sympy").cos(tj * __import__("sympy").log(1 + ti)),
                "Cos[tj Log[1 + ti]]",
            ),
            FeatureSpec(
                "sin((t_i-t_j)*log(1+Delta))",
                lambda ti, tj, d: np.sin((ti - tj) * np.log1p(d)),
                lambda ti, tj, d: __import__("sympy").sin((ti - tj) * __import__("sympy").log(1 + d)),
                "Sin[(ti - tj) Log[1 + Delta]]",
            ),
        ]
    )
    return specs


def sample_pairs(n: int, max_pairs: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return sampled row/column index pairs without materializing N^2 when large."""
    if max_pairs <= 0:
        raise ValueError("--max-pairs must be positive")
    total = n * n
    rng = np.random.default_rng(seed)
    if total <= max_pairs:
        flat = np.arange(total, dtype=np.int64)
    else:
        flat = rng.choice(total, size=max_pairs, replace=False)
    return flat // n, flat % n


def design_matrix_for_pairs(
    t: np.ndarray, rows: np.ndarray, cols: np.ndarray, specs: Sequence[FeatureSpec]
) -> np.ndarray:
    """Build feature matrix for sampled pairs."""
    ti = t[rows]
    tj = t[cols]
    delta = np.abs(ti - tj)
    X = np.column_stack([spec.evaluator(ti, tj, delta) for spec in specs]).astype(np.float64)
    finite = np.all(np.isfinite(X), axis=0)
    if not np.all(finite):
        names = [specs[i].name for i, ok in enumerate(finite) if not ok]
        raise FloatingPointError(f"feature columns contain NaN/Inf: {names}")
    return X


def compute_feature_row(ti: float, tj: float, fourier_k: int, eps: float = 1e-6) -> List[float]:
    """Compute one picklable real feature row in the same order as build_feature_specs()."""
    d = abs(ti - tj)
    feats = [
        1.0,
        d,
        d * d,
        math.log1p(d),
        1.0 / (1.0 + d),
        math.exp(-((ti - tj) ** 2)),
    ]

    for k in range(1, fourier_k + 1):
        feats.append(math.cos(k * d))
        feats.append(math.sin(k * d))

    feats.extend(
        [
            1.0 / (eps + d),
            1.0 / (eps + d * d),
            math.cos(ti * math.log1p(tj)),
            math.cos(tj * math.log1p(ti)),
            math.sin((ti - tj) * math.log1p(d)),
        ]
    )
    return feats


def feature_batch_worker(
    batch: Sequence[Tuple[int, int, int]],
    t: np.ndarray,
    H_values: np.ndarray,
    fourier_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sampled feature rows and complex targets for a batch of (idx, i, j)."""
    X_batch: List[List[float]] = []
    y_batch: List[complex] = []

    for _, i, j in batch:
        ti = float(t[i])
        tj = float(t[j])
        X_batch.append(compute_feature_row(ti, tj, fourier_k))
        y_batch.append(complex(H_values[i, j]))

    return np.asarray(X_batch, dtype=np.float64), np.asarray(y_batch, dtype=np.complex128)


def build_feature_matrix_parallel(
    pairs: Sequence[Tuple[int, int, int]],
    t: np.ndarray,
    H_values: np.ndarray,
    fourier_k: int,
    workers: int = 4,
    batch_size: int = 50_000,
    progress: bool = True,
    use_multiprocessing: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sampled feature matrix with optional spawn-safe multiprocessing."""
    batches = list(make_batches(pairs, batch_size))
    if not batches:
        raise ValueError("no sampled pairs available for feature matrix construction")

    if (not use_multiprocessing) or workers <= 1:
        X_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        eta = ETATracker(len(batches), name="feature_batches", enabled=progress and not HAS_TQDM)
        iterator = progress_iter(
            enumerate(batches),
            total=len(batches),
            desc="feature_build",
            enabled=progress,
        )
        for bi, batch in iterator:
            Xb, yb = feature_batch_worker(batch, t, H_values, fourier_k)
            X_parts.append(Xb)
            y_parts.append(yb)
            eta.update(bi + 1)
        return np.vstack(X_parts), np.concatenate(y_parts)

    try:
        ctx = mp.get_context("spawn")
        worker_fn = partial(
            feature_batch_worker,
            t=t,
            H_values=H_values,
            fourier_k=fourier_k,
        )

        X_parts = []
        y_parts = []
        with ctx.Pool(processes=workers) as pool:
            iterator = pool.imap(worker_fn, batches)
            if progress and HAS_TQDM and tqdm is not None:
                iterator = tqdm(iterator, total=len(batches), desc="feature_build_mp")
            eta = ETATracker(len(batches), name="feature_build_mp", enabled=progress and not HAS_TQDM)
            for bi, (Xb, yb) in enumerate(iterator):
                X_parts.append(Xb)
                y_parts.append(yb)
                eta.update(bi + 1)
        return np.vstack(X_parts), np.concatenate(y_parts)
    except Exception as exc:
        print(f"[WARN] multiprocessing feature build failed: {exc}")
        print("[WARN] falling back to single-process feature build")
        return build_feature_matrix_parallel(
            pairs=pairs,
            t=t,
            H_values=H_values,
            fourier_k=fourier_k,
            workers=1,
            batch_size=batch_size,
            progress=progress,
            use_multiprocessing=False,
        )


def _standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center and scale feature columns for stable regression."""
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < EPS] = 1.0
    return (X - mean) / scale, mean, scale


def _unscale_coefficients(beta: np.ndarray, intercept: complex, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Map standardized linear model coefficients back to raw feature coefficients."""
    coeff = beta / scale
    coeff = coeff.astype(np.complex128, copy=False)
    coeff[0] += intercept - np.sum(mean * coeff)
    return coeff


def _fit_lasso_coefficients(X: np.ndarray, y: np.ndarray, seed: int) -> Optional[np.ndarray]:
    """Fit a complex linear model using sklearn LassoCV when available."""
    try:
        from sklearn.linear_model import LassoCV
    except Exception:
        return None

    Xs, mean, scale = _standardize_features(X)
    cv = min(5, max(2, X.shape[0] // 2000))
    kwargs = {
        "cv": cv,
        "fit_intercept": True,
        "random_state": seed,
        "max_iter": 20_000,
        "n_jobs": None,
    }
    real_model = LassoCV(**kwargs).fit(Xs, y.real)
    imag_model = LassoCV(**kwargs).fit(Xs, y.imag)
    beta = real_model.coef_.astype(np.complex128) + 1j * imag_model.coef_.astype(np.complex128)
    intercept = complex(float(real_model.intercept_), float(imag_model.intercept_))
    return _unscale_coefficients(beta, intercept, mean, scale)


def sklearn_lasso_available() -> bool:
    """Return whether sklearn LassoCV can be imported for sparse regression."""
    try:
        from sklearn.linear_model import LassoCV  # noqa: F401
    except Exception:
        return False
    return True


def _least_squares_coefficients(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit dense complex least squares coefficients as a dependency-free fallback."""
    coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.asarray(coeff, dtype=np.complex128)


def _compact_refit(
    X: np.ndarray,
    y: np.ndarray,
    seed_coefficients: np.ndarray,
    alpha_sparsity: float,
    max_terms: int = 20,
) -> RegressionResult:
    """Select a compact support by L0-scored refits over ranked candidate supports."""
    p = X.shape[1]
    ranking = list(np.argsort(-np.abs(seed_coefficients)))
    if 0 in ranking:
        ranking.remove(0)
    ranking = [0] + ranking

    candidate_sizes = sorted(set(range(1, min(max_terms, p) + 1)) | {min(p, max(1, np.count_nonzero(seed_coefficients)))})
    best: Optional[RegressionResult] = None

    for size in candidate_sizes:
        support = np.array(ranking[: min(size, p)], dtype=int)
        coeff_support, *_ = np.linalg.lstsq(X[:, support], y, rcond=None)
        coeff = np.zeros(p, dtype=np.complex128)
        coeff[support] = coeff_support
        pred = X @ coeff
        mse = float(np.mean(np.abs(pred - y) ** 2))
        nonzero = int(np.count_nonzero(np.abs(coeff) > 1e-10))
        score = float(mse + alpha_sparsity * nonzero)
        result = RegressionResult(coefficients=coeff, selected=np.flatnonzero(np.abs(coeff) > 1e-10), mse=mse, score=score, method="")
        if best is None or result.score < best.score:
            best = result

    if best is None:
        raise RuntimeError("sparse regression failed to produce a candidate model")
    return best


def fit_sparse_regression(X: np.ndarray, y: np.ndarray, alpha_sparsity: float, seed: int) -> RegressionResult:
    """Fit sparse symbolic regression with sklearn LassoCV or a least-squares fallback."""
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if not np.all(finite):
        X = X[finite]
        y = y[finite]
    if X.shape[0] < X.shape[1]:
        raise ValueError(f"not enough finite samples for regression: {X.shape[0]} rows, {X.shape[1]} features")

    seed_coeff = _fit_lasso_coefficients(X, y, seed)
    method = "LassoCV + L0 compact refit"
    if seed_coeff is None or not np.all(np.isfinite(seed_coeff)):
        seed_coeff = _least_squares_coefficients(X, y)
        method = "least squares + L0 compact refit"

    result = _compact_refit(X, y, seed_coeff, alpha_sparsity)
    result.method = method
    return result


def _format_complex(c: complex, precision: int = 12) -> str:
    """Format complex coefficients compactly for text output."""
    c = complex(c)
    real = 0.0 if abs(c.real) < 1e-14 else c.real
    imag = 0.0 if abs(c.imag) < 1e-14 else c.imag
    if imag == 0.0:
        return f"{real:.{precision}g}"
    if real == 0.0:
        return f"{imag:.{precision}g}*I"
    sign = "+" if imag >= 0 else "-"
    return f"({real:.{precision}g} {sign} {abs(imag):.{precision}g}*I)"


def symbolic_complexity(expr: Any) -> Dict[str, int]:
    """Return conservative SymPy expression complexity diagnostics."""
    if sympy is None:
        return {"ops": 10**9, "terms": 10**9}
    try:
        n_ops = int(sympy.count_ops(expr))
    except Exception:
        n_ops = 10**9

    try:
        n_terms = len(sympy.Add.make_args(sympy.expand(expr)))
    except Exception:
        n_terms = 10**9

    return {
        "ops": n_ops,
        "terms": n_terms,
    }


def _call_with_timeout(fn: Callable[[Any], Any], expr: Any, timeout: float, label: str) -> Any:
    """Call a SymPy transform with a Unix timer when available."""
    if timeout <= 0 or not hasattr(signal, "setitimer"):
        return fn(expr)

    def _raise_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise TimeoutError(f"{label} exceeded {timeout:.1f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        return fn(expr)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def safe_symbolic_simplify(
    expr: Any,
    ti: Any,
    tj: Any,
    mode: str = "safe",
    max_ops: int = 250,
    max_terms: int = 12,
    timeout: float = 20.0,
) -> Tuple[Any, Dict[str, Any]]:
    """Simplify a SymPy expression using complexity guards and timed operations."""
    if sympy is None:
        report = {
            "mode": mode,
            "before": {"ops": None, "terms": None},
            "after": {"ops": None, "terms": None},
            "skipped": True,
            "reason": "sympy_unavailable",
        }
        return expr, report

    info_before = symbolic_complexity(expr)
    print(f"[sympy] complexity_before={info_before}")

    if mode == "none":
        return expr, {
            "mode": mode,
            "before": info_before,
            "after": info_before,
            "skipped": True,
            "reason": "sympy_mode_none",
        }

    if info_before["terms"] > max_terms or info_before["ops"] > max_ops:
        print("[sympy][WARN] expression too complex; using safe fast simplification only")
        expr2 = expr

        try:
            expr2 = _call_with_timeout(lambda x: sympy.collect(x, [ti, tj], evaluate=True), expr2, timeout, "collect")
        except Exception as exc:
            print(f"[sympy][WARN] collect failed: {exc}")

        try:
            expr2 = _call_with_timeout(sympy.factor_terms, expr2, timeout, "factor_terms")
        except Exception as exc:
            print(f"[sympy][WARN] factor_terms failed: {exc}")

        info_after = symbolic_complexity(expr2)
        return expr2, {
            "mode": mode,
            "before": info_before,
            "after": info_after,
            "skipped": True,
            "reason": "complexity_guard",
        }

    if mode == "fast":
        expr2 = expr

        try:
            expr2 = _call_with_timeout(sympy.expand, expr2, timeout, "expand")
        except Exception as exc:
            print(f"[sympy][WARN] expand failed: {exc}")

        try:
            expr2 = _call_with_timeout(lambda x: sympy.collect(x, [ti, tj], evaluate=True), expr2, timeout, "collect")
        except Exception as exc:
            print(f"[sympy][WARN] collect failed: {exc}")

        try:
            expr2 = _call_with_timeout(sympy.factor_terms, expr2, timeout, "factor_terms")
        except Exception as exc:
            print(f"[sympy][WARN] factor_terms failed: {exc}")

        info_after = symbolic_complexity(expr2)
        return expr2, {
            "mode": mode,
            "before": info_before,
            "after": info_after,
            "skipped": False,
            "reason": None,
        }

    if mode == "safe":
        expr2 = expr

        for fn_name, fn in [
            ("factor_terms", sympy.factor_terms),
            ("trigsimp", sympy.trigsimp),
            ("powsimp", sympy.powsimp),
        ]:
            try:
                before_ops = symbolic_complexity(expr2)["ops"]
                candidate = _call_with_timeout(fn, expr2, timeout, fn_name)
                if symbolic_complexity(candidate)["ops"] <= before_ops:
                    expr2 = candidate
                    print(f"[sympy] accepted {fn_name}")
                else:
                    print(f"[sympy] rejected {fn_name}: complexity increased")
            except Exception as exc:
                print(f"[sympy][WARN] {fn_name} failed: {exc}")

        info_after = symbolic_complexity(expr2)
        return expr2, {
            "mode": mode,
            "before": info_before,
            "after": info_after,
            "skipped": False,
            "reason": None,
        }

    if mode == "full":
        print("[sympy][WARN] full simplify may be slow")
        try:
            expr2 = _call_with_timeout(sympy.simplify, expr, timeout, "simplify")
            skipped = False
            reason = None
        except Exception as exc:
            print(f"[sympy][WARN] full simplify failed: {exc}")
            expr2 = expr
            skipped = True
            reason = str(exc)
        info_after = symbolic_complexity(expr2)

        return expr2, {
            "mode": mode,
            "before": info_before,
            "after": info_after,
            "skipped": skipped,
            "reason": reason,
        }

    return expr, {
        "mode": mode,
        "before": info_before,
        "after": info_before,
        "skipped": True,
        "reason": "unknown_mode",
    }


def build_symbolic_expression(
    specs: Sequence[FeatureSpec],
    coefficients: np.ndarray,
    selected_indices: Optional[np.ndarray] = None,
    sympy_mode: str = "safe",
    max_symbolic_ops: int = 250,
    max_symbolic_terms: int = 12,
    simplify_timeout: float = 20.0,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """Build SymPy text, LaTeX text, and Wolfram input for the selected model."""
    selected = np.asarray(selected_indices, dtype=int) if selected_indices is not None else np.flatnonzero(np.abs(coefficients) > 1e-10)
    wolfram_terms = []
    for idx in selected:
        coeff = complex(coefficients[idx])
        coeff_w = _format_complex(coeff).replace("*I", " I")
        wolfram_terms.append(f"({coeff_w})*({specs[idx].wolfram})")
    wolfram_input = " + ".join(wolfram_terms) if wolfram_terms else "0"

    if sympy is None:
        text_terms = [f"{_format_complex(coefficients[i])} * {specs[i].name}" for i in selected]
        expression = " + ".join(text_terms) if text_terms else "0"
        sympy_report = {
            "mode": sympy_mode,
            "before": {"ops": None, "terms": None},
            "after": {"ops": None, "terms": None},
            "skipped": True,
            "reason": "sympy_unavailable",
        }
        return expression, expression, wolfram_input, sympy_report

    ti, tj = sympy.symbols("t_i t_j", real=True)
    delta = sympy.Abs(ti - tj)
    expr = 0
    for idx in selected:
        coeff = complex(coefficients[idx])
        coeff_sym = sympy.Float(coeff.real, 16) + sympy.I * sympy.Float(coeff.imag, 16)
        expr += coeff_sym * specs[idx].sympy_builder(ti, tj, delta)
    simplify_start = time.time()
    expr, sympy_report = safe_symbolic_simplify(
        expr,
        ti,
        tj,
        mode=sympy_mode,
        max_ops=max_symbolic_ops,
        max_terms=max_symbolic_terms,
        timeout=simplify_timeout,
    )
    print(f"[sympy] report={sympy_report}")
    print(f"[V11.2] simplify_time={time.time() - simplify_start:.2f}s")
    try:
        latex_text = sympy.latex(expr)
    except Exception:
        latex_text = str(expr)
    return str(expr), latex_text, wolfram_input, sympy_report


def maybe_wolfram_simplify(wolfram_input: str) -> Optional[str]:
    """Run optional Wolfram Engine simplification if wolframscript is installed."""
    executable = shutil.which("wolframscript")
    if executable is None:
        return None
    command = (
        f"expr = {wolfram_input}; "
        "Print[InputForm[FullSimplify[FunctionExpand[expr], "
        "Assumptions -> 0 <= ti <= 1 && 0 <= tj <= 1 && Delta == Abs[ti - tj]]]]"
    )
    try:
        proc = subprocess.run(
            [executable, "-code", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        return f"Wolfram simplification skipped: {exc}"
    if proc.returncode != 0:
        return f"Wolfram simplification failed: {proc.stderr.strip()}"
    return proc.stdout.strip()


def evaluate_feature_block(
    t: np.ndarray, row_start: int, row_stop: int, specs: Sequence[FeatureSpec]
) -> np.ndarray:
    """Evaluate all features for rows [row_start, row_stop) against the full grid."""
    ti = t[row_start:row_stop, None]
    tj = t[None, :]
    delta = np.abs(ti - tj)
    return np.stack([spec.evaluator(ti, tj, delta) for spec in specs], axis=-1)


def _pair_batch_generator(n: int, batch_size: int) -> Iterator[List[Tuple[int, int]]]:
    """Yield full-grid (i, j) pair batches without materializing N^2 pairs."""
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    batch: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            batch.append((i, j))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def kernel_eval_batch_worker(
    batch: Sequence[Tuple[int, int]],
    t: np.ndarray,
    coeffs: np.ndarray,
    selected_feature_indices: np.ndarray,
    fourier_k: int,
) -> List[Tuple[int, int, complex]]:
    """Evaluate selected symbolic features for one full-kernel pair batch."""
    rows: List[Tuple[int, int, complex]] = []
    selected = np.asarray(selected_feature_indices, dtype=int)
    selected_coeffs = np.asarray(coeffs, dtype=np.complex128)

    for i, j in batch:
        feats = compute_feature_row(float(t[i]), float(t[j]), fourier_k)
        val = 0.0 + 0.0j
        for c, idx in zip(selected_coeffs, selected):
            val += complex(c) * feats[int(idx)]
        rows.append((i, j, val))
    return rows


def _hermitianize_memmap(out: np.ndarray, block_rows: int) -> None:
    """Enforce Hermitian symmetry on a dense array or memmap in blocks."""
    n = out.shape[0]
    blocks = [(start, min(n, start + block_rows)) for start in range(0, n, block_rows)]
    for bi, (r0, r1) in enumerate(blocks):
        for c0, c1 in blocks[bi:]:
            upper = np.array(out[r0:r1, c0:c1], copy=True)
            if r0 == c0 and r1 == c1:
                sym = 0.5 * (upper + upper.conj().T)
                out[r0:r1, c0:c1] = sym
            else:
                lower = np.array(out[c0:c1, r0:r1], copy=True)
                sym = 0.5 * (upper + lower.conj().T)
                out[r0:r1, c0:c1] = sym
                out[c0:c1, r0:r1] = sym.conj().T
    if hasattr(out, "flush"):
        out.flush()


def evaluate_kernel_parallel(
    n: int,
    t: np.ndarray,
    coeffs: np.ndarray,
    selected_feature_indices: np.ndarray,
    fourier_k: int,
    workers: int = 4,
    batch_size: int = 50_000,
    progress: bool = True,
    use_multiprocessing: bool = True,
    out_path: Optional[Path] = None,
) -> np.ndarray:
    """Evaluate the full symbolic kernel with optional multiprocessing and progress."""
    if n <= 0:
        raise ValueError("kernel dimension must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    total_batches = int(math.ceil((n * n) / batch_size))
    if out_path is None:
        K: np.ndarray = np.zeros((n, n), dtype=np.complex128)
    else:
        K = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.complex128, shape=(n, n))

    if (not use_multiprocessing) or workers <= 1:
        eta = ETATracker(total_batches, name="kernel_eval", enabled=progress and not HAS_TQDM)
        iterator = progress_iter(
            enumerate(_pair_batch_generator(n, batch_size)),
            total=total_batches,
            desc="kernel_eval",
            enabled=progress,
        )
        for bi, batch in iterator:
            rows = kernel_eval_batch_worker(batch, t, coeffs, selected_feature_indices, fourier_k)
            for i, j, val in rows:
                K[i, j] = val
            eta.update(bi + 1)
        _hermitianize_memmap(K, max(1, min(n, batch_size // max(1, n))))
        return K

    try:
        ctx = mp.get_context("spawn")
        worker_fn = partial(
            kernel_eval_batch_worker,
            t=t,
            coeffs=coeffs,
            selected_feature_indices=selected_feature_indices,
            fourier_k=fourier_k,
        )
        with ctx.Pool(processes=workers) as pool:
            iterator = pool.imap(worker_fn, _pair_batch_generator(n, batch_size))
            if progress and HAS_TQDM and tqdm is not None:
                iterator = tqdm(iterator, total=total_batches, desc="kernel_eval_mp")
            eta = ETATracker(total_batches, name="kernel_eval_mp", enabled=progress and not HAS_TQDM)
            for bi, rows in enumerate(iterator):
                for i, j, val in rows:
                    K[i, j] = val
                eta.update(bi + 1)
        _hermitianize_memmap(K, max(1, min(n, batch_size // max(1, n))))
        return K
    except Exception as exc:
        print(f"[WARN] multiprocessing kernel eval failed: {exc}")
        print("[WARN] falling back to single-process kernel eval")
        return evaluate_kernel_parallel(
            n=n,
            t=t,
            coeffs=coeffs,
            selected_feature_indices=selected_feature_indices,
            fourier_k=fourier_k,
            workers=1,
            batch_size=batch_size,
            progress=progress,
            use_multiprocessing=False,
            out_path=out_path,
        )


def reconstruct_operator(
    t: np.ndarray,
    specs: Sequence[FeatureSpec],
    coefficients: np.ndarray,
    out_path: Path,
    block_rows: Optional[int] = None,
) -> np.memmap:
    """Evaluate K(t_i,t_j) on the full grid and save a Hermitian .npy array."""
    n = t.size
    if block_rows is None:
        target_entries = max(1, 2_000_000 // max(1, len(specs)))
        block_rows = max(1, min(n, target_entries // max(1, n)))

    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.complex128, shape=(n, n))
    for start in range(0, n, block_rows):
        stop = min(n, start + block_rows)
        F = evaluate_feature_block(t, start, stop, specs)
        out[start:stop, :] = np.tensordot(F, coefficients, axes=([-1], [0]))
    out.flush()

    # Enforce the analytic reconstruction as an operator-level Hermitian candidate.
    _hermitianize_memmap(out, block_rows)
    return out


def relative_frobenius_error(A: np.ndarray, B: np.ndarray, block_rows: int = 512) -> float:
    """Compute ||A-B||_F / ||A||_F in blocks."""
    n = A.shape[0]
    num = 0.0
    den = 0.0
    for start in range(0, n, block_rows):
        stop = min(n, start + block_rows)
        diff = A[start:stop, :] - B[start:stop, :]
        num += float(np.sum(np.abs(diff) ** 2))
        den += float(np.sum(np.abs(A[start:stop, :]) ** 2))
    return float(math.sqrt(num) / max(math.sqrt(den), EPS))


def compute_eigenvalues(H: np.ndarray, label: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute eigenvalues exactly for moderate N, or on a deterministic principal sketch for large N."""
    n = H.shape[0]
    info: Dict[str, Any] = {"label": label, "n": n, "mode": "dense"}
    if n <= MAX_DENSE_EIG_N:
        eig = np.linalg.eigvalsh(np.asarray(H, dtype=np.complex128))
        return np.sort(np.real(eig)), info

    idx = np.linspace(0, n - 1, MAX_DENSE_EIG_N, dtype=int)
    sketch = np.asarray(H[np.ix_(idx, idx)], dtype=np.complex128)
    eig = np.linalg.eigvalsh(sketch)
    info.update({"mode": "principal_submatrix_sketch", "sketch_n": int(idx.size)})
    return np.sort(np.real(eig)), info


def spectral_metrics(H_true: np.ndarray, H_fit: np.ndarray, first_k: int = DEFAULT_FIRST_K) -> Dict[str, Any]:
    """Compute spectral relative and first-k errors for true and reconstructed operators."""
    eig_true, true_info = compute_eigenvalues(H_true, "H_herm")
    eig_fit, fit_info = compute_eigenvalues(H_fit, "K_fit")
    k_all = min(eig_true.size, eig_fit.size)
    if k_all == 0:
        raise ValueError("no eigenvalues available for spectral validation")
    spectral_error = float(
        np.linalg.norm(eig_true[:k_all] - eig_fit[:k_all]) / max(float(np.linalg.norm(eig_true[:k_all])), EPS)
    )
    k_first = min(first_k, k_all)
    first_error = float(
        np.linalg.norm(eig_true[:k_first] - eig_fit[:k_first]) / max(float(np.linalg.norm(eig_true[:k_first])), EPS)
    )
    return {
        "spectral_relative_error": spectral_error,
        "first_k_spectral_error": first_error,
        "first_k": int(k_first),
        "eigenvalue_range_true": [float(eig_true[0]), float(eig_true[-1])],
        "eigenvalue_range_fit": [float(eig_fit[0]), float(eig_fit[-1])],
        "eigenvalue_std_true": float(np.std(eig_true)),
        "eigenvalue_std_fit": float(np.std(eig_fit)),
        "eigen_solver_true": true_info,
        "eigen_solver_fit": fit_info,
    }


def _spacing(values: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(values, dtype=float).reshape(-1))
    return np.diff(values) if values.size > 1 else np.asarray([], dtype=float)


def _load_zeta_csv(path: Path) -> np.ndarray:
    rows: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                try:
                    rows.append(float(cell))
                    break
                except ValueError:
                    continue
    zeros = np.asarray(rows, dtype=float)
    zeros = np.sort(np.abs(zeros[np.isfinite(zeros)]))
    return zeros[zeros > 0]


def find_zeta_zeros_csv(operator_path: Path, out_dir: Path) -> Optional[Path]:
    """Find a nearby zeta_zeros.csv without recursively scanning large trees."""
    candidates = [
        out_dir / "zeta_zeros.csv",
        out_dir.parent / "zeta_zeros.csv",
        operator_path.parent / "zeta_zeros.csv",
        Path("zeta_zeros.csv"),
        Path("runs") / "zeta_zeros.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def zeta_diagnostics(eig_fit: np.ndarray, zeta_csv: Optional[Path]) -> Dict[str, Any]:
    """Compare reconstructed spectrum with zeta zero spacing when a CSV is available."""
    if zeta_csv is None:
        return {"zeta_zeros_csv": None, "zeta_diagnostics_available": False}
    zeros = _load_zeta_csv(zeta_csv)
    if zeros.size < 2 or eig_fit.size < 2:
        return {"zeta_zeros_csv": str(zeta_csv), "zeta_diagnostics_available": False}
    k = min(zeros.size, eig_fit.size)
    zs = _spacing(zeros[:k])
    es = _spacing(eig_fit[:k])
    nearest = np.min(np.abs(eig_fit[:k, None] - zeros[:k][None, :]), axis=0)
    return {
        "zeta_zeros_csv": str(zeta_csv),
        "zeta_diagnostics_available": True,
        "zeta_k": int(k),
        "spacing_mean_error": float(abs(np.mean(es) - np.mean(zs))),
        "spacing_std_error": float(abs(np.std(es) - np.std(zs))),
        "nearest_spectrum_to_zeta_error": float(np.mean(nearest)),
    }


def compute_symbolic_reward(H: np.ndarray, metrics: Dict[str, Any], config: Dict[str, float]) -> float:
    """Compute the RL reward contribution for symbolic operator quality."""
    del H
    kernel_error = float(metrics.get("kernel_relative_error", 0.0) or 0.0)
    spectral_error = float(metrics.get("spectral_relative_error", 0.0) or 0.0)
    terms = float(metrics.get("num_terms", 0.0) or 0.0)
    hermitian_error = float(metrics.get("hermitian_error", 0.0) or 0.0)
    return float(
        -kernel_error
        - float(config.get("lambda_spec", 1.0)) * spectral_error
        - float(config.get("lambda_complexity", 0.01)) * terms
        - float(config.get("lambda_herm", 10.0)) * hermitian_error
    )


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text with parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Run V11.2 symbolic reconstruction from operator load through RL feedback."""
    stage_logger.reset()
    operator_path = Path(args.operator)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[V11.2] Symbolic reconstruction acceleration enabled")
    print(f"[V11.2] workers={args.workers}, batch_size={args.batch_size}, progress={args.progress}")
    print(f"[V11.2] multiprocessing={not args.no_multiprocessing}")

    stage_logger.start("load_operator")
    H = load_operator(operator_path)
    n = H.shape[0]
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    print(f"[V11.2] operator_shape=({n}, {n}), dtype={H.dtype}")
    stage_logger.end()

    stage_logger.start("hermitian_projection")
    H_herm, hermitian_error = hermitian_projection(H)
    print(f"[V11.2] hermitian_error={hermitian_error:.6e}")
    stage_logger.end()

    stage_logger.start("pair_sampling")
    specs = build_feature_specs(args.fourier_k)
    rows, cols = sample_pairs(n, args.max_pairs, args.seed)
    pairs = [(idx, int(i), int(j)) for idx, (i, j) in enumerate(zip(rows, cols))]
    print(f"[V11.2] pairs: {len(pairs)}")
    print(f"[V11.2] feature_count: {len(specs)}")
    stage_logger.end()

    stage_logger.start("feature_build")
    print("[V11.2] >>> entering FEATURE BUILD")
    print(f"[V11.2] building feature matrix with {len(pairs)} pairs...")
    X, y = build_feature_matrix_parallel(
        pairs=pairs,
        t=t,
        H_values=H_herm,
        fourier_k=int(args.fourier_k),
        workers=int(args.workers),
        batch_size=int(args.batch_size),
        progress=bool(args.progress),
        use_multiprocessing=not bool(args.no_multiprocessing),
    )
    print(f"[V11.2] feature matrix shape: {X.shape}")
    stage_logger.end()

    stage_logger.start("sparse_regression")
    print("[V11.2] >>> entering REGRESSION")
    backend = "sklearn LassoCV" if sklearn_lasso_available() else "fallback least squares + L0 compact refit"
    print(f"[V11.2] regression backend: {backend}")
    t0 = time.time()
    regression = fit_sparse_regression(X, y, float(args.alpha_sparsity), int(args.seed))
    regression_time = time.time() - t0
    selected_terms = [specs[i].name for i in regression.selected]
    print(f"[V11.2] regression_time={regression_time:.2f}s")
    print(f"[V11.2] selected_terms={len(selected_terms)}")
    print(f"[V11.2] regression_method={regression.method}")
    stage_logger.end()

    stage_logger.start("sympy_build")
    print("[V11.2] >>> entering SYMPY")
    print("[V11.2] building SymPy expression...")
    symbolic_selected_indices = regression.selected.astype(int)
    if len(symbolic_selected_indices) > int(args.max_symbolic_terms):
        print(
            f"[sympy][WARN] selected_terms={len(symbolic_selected_indices)} exceeds "
            f"max_symbolic_terms={args.max_symbolic_terms}; truncating by coefficient magnitude"
        )
        order = np.argsort(np.abs(regression.coefficients[symbolic_selected_indices]))[::-1]
        symbolic_selected_indices = symbolic_selected_indices[order[: int(args.max_symbolic_terms)]]
    print(f"[sympy] symbolic_terms={len(symbolic_selected_indices)}")
    sympy_text, latex_text, wolfram_input, sympy_report = build_symbolic_expression(
        specs,
        regression.coefficients,
        selected_indices=symbolic_selected_indices,
        sympy_mode=str(args.sympy_mode),
        max_symbolic_ops=int(args.max_symbolic_ops),
        max_symbolic_terms=int(args.max_symbolic_terms),
        simplify_timeout=float(args.simplify_timeout),
    )
    formula_text = f"{TITLE}\n\nK(t_i, t_j) = {sympy_text}\n"

    write_text(out_dir / "symbolic_kernel_candidate_v2_sympy.txt", formula_text)
    write_text(out_dir / "symbolic_kernel_candidate_v2_latex.tex", f"% {TITLE}\n{latex_text}\n")
    with (out_dir / "symbolic_simplification_report_v2.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(sympy_report), f, indent=2)
    wolfram_output = maybe_wolfram_simplify(wolfram_input)
    if wolfram_output is not None:
        write_text(out_dir / "symbolic_kernel_candidate_v2_wolfram.txt", f"{TITLE}\n\n{wolfram_output}\n")
    stage_logger.end()

    stage_logger.start("kernel_reconstruction")
    print("[V11.2] >>> entering KERNEL RECONSTRUCTION")
    print(f"[V11.2] reconstructing full operator (N={n}, N^2={n * n})")
    fit_path = out_dir / "symbolic_operator_fit_v2.npy"
    selected_feature_indices = regression.selected.astype(int)
    K_fit = evaluate_kernel_parallel(
        n=n,
        t=t,
        coeffs=regression.coefficients[selected_feature_indices],
        selected_feature_indices=selected_feature_indices,
        fourier_k=int(args.fourier_k),
        workers=int(args.workers),
        batch_size=int(args.batch_size),
        progress=bool(args.progress),
        use_multiprocessing=not bool(args.no_multiprocessing),
        out_path=fit_path,
    )
    kernel_error = relative_frobenius_error(H_herm, K_fit)
    print(f"[V11.2] kernel_relative_error={kernel_error:.6f}")
    stage_logger.end()

    stage_logger.start("spectral_validation")
    spec = spectral_metrics(H_herm, K_fit)
    true_range = spec["eigenvalue_range_true"][1] - spec["eigenvalue_range_true"][0]
    fit_range = spec["eigenvalue_range_fit"][1] - spec["eigenvalue_range_fit"][0]
    print(f"[V11.2] eig_range_true={true_range:.3f}")
    print(f"[V11.2] eig_range_fit={fit_range:.3f}")
    print(f"[V11.2] spectral_error={spec['spectral_relative_error']:.6f}")
    eig_fit, _ = compute_eigenvalues(K_fit, "K_fit_for_zeta")
    zeta = zeta_diagnostics(eig_fit, find_zeta_zeros_csv(operator_path, out_dir))
    stage_logger.end()

    selected_coefficients = {
        specs[i].name: {"real": float(regression.coefficients[i].real), "imag": float(regression.coefficients[i].imag)}
        for i in regression.selected
    }
    metrics: Dict[str, Any] = {
        "kernel_relative_error": kernel_error,
        "hermitian_error": hermitian_error,
        "num_terms": int(regression.selected.size),
        "regression_sample_mse": regression.mse,
        "regression_score": regression.score,
        **spec,
        **zeta,
    }
    config = {
        "lambda_spec": float(args.lambda_spec),
        "lambda_complexity": float(args.lambda_complexity),
        "lambda_herm": float(args.lambda_herm),
    }
    stage_logger.start("rl_feedback")
    reward = compute_symbolic_reward(H_herm, metrics, config)
    metrics["symbolic_quality_score"] = reward
    metrics["suggested_reward_bonus"] = reward
    print(f"[V11.2] kernel_error={kernel_error:.6f}")
    print(f"[V11.2] spectral_error={spec['spectral_relative_error']:.6f}")
    print(f"[V11.2] hermitian_error={hermitian_error:.6e}")
    print(f"[V11.2] reward={reward:.6f}")

    report: Dict[str, Any] = {
        "version": "V11.2 Symbolic Operator Reconstruction Pipeline (Sparse + RL Feedback)",
        "label": TITLE,
        "operator": str(operator_path),
        "operator_shape": [int(n), int(n)],
        "grid": "linspace(0,1,N)",
        "fourier_k": int(args.fourier_k),
        "alpha_sparsity": float(args.alpha_sparsity),
        "max_pairs": int(args.max_pairs),
        "sampled_pairs": int(rows.size),
        "regression_method": regression.method,
        "selected_terms": selected_terms,
        "coefficients": selected_coefficients,
        "formula": sympy_text,
        "sympy_report": sympy_report,
        "metrics": metrics,
        "outputs": {
            "sympy": str(out_dir / "symbolic_kernel_candidate_v2_sympy.txt"),
            "latex": str(out_dir / "symbolic_kernel_candidate_v2_latex.tex"),
            "wolfram": str(out_dir / "symbolic_kernel_candidate_v2_wolfram.txt") if wolfram_output is not None else None,
            "operator_fit": str(fit_path),
            "feedback": str(out_dir / "symbolic_rl_feedback_v2.json"),
            "report": str(out_dir / "report.json"),
            "sympy_report": str(out_dir / "symbolic_simplification_report_v2.json"),
        },
        "note": TITLE,
    }

    feedback = {
        "label": TITLE,
        "symbolic_quality_score": reward,
        "suggested_reward_bonus": reward,
        "selected_terms": selected_terms,
        "coefficients": selected_coefficients,
        "sympy_report": sympy_report,
        "metrics": metrics,
        "config": config,
    }

    with (out_dir / "symbolic_rl_feedback_v2.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(feedback), f, indent=2)
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)
    stage_logger.end()

    print(TITLE)
    print(f"hermitian error: {hermitian_error:.6e}")
    print(f"number of terms: {regression.selected.size}")
    print(f"formula: K(t_i, t_j) = {sympy_text}")
    print(f"kernel error: {kernel_error:.6e}")
    print(f"spectral error: {spec['spectral_relative_error']:.6e}")
    print(f"RL score: {reward:.6e}")
    print("\n=== V11.2 ACCELERATION SUMMARY ===")
    print(f"workers: {args.workers}")
    print(f"batch_size: {args.batch_size}")
    print(f"progress: {args.progress}")
    print(f"multiprocessing: {not args.no_multiprocessing}")
    print("\n=== V11.2 FULL PIPELINE SUMMARY ===")
    print(f"Total runtime: {time.time() - stage_logger.global_start:.2f}s")
    print(f"Output dir: {out_dir}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    try:
        run_pipeline(args)
    except Exception as exc:
        raise RuntimeError(f"V11.2 symbolic reconstruction failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
