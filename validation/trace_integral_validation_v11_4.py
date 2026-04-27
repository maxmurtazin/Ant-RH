from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import sympy

    HAS_SYMPY = True
except Exception:
    sympy = None
    HAS_SYMPY = False


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = "V11.4 Trace-Formula / Integral-Operator Validation"
NOTE = "Trace/integral-operator diagnostic; not proof of RH."
OPERATOR_FORM = "(Hf)(t)=∫K(t,s)f(s)ds"
EPS = 1e-12


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse V11.4 CLI arguments."""
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--formula", required=True, help="Symbolic kernel formula text file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--learned-operator", default=None, help="Optional learned operator .npy")
    parser.add_argument("--zeta-zeros", default=None, help="Optional zeta zeros CSV")
    parser.add_argument("--n-grid", type=int, default=512, help="Integral grid size")
    parser.add_argument("--quad", choices=["trapezoid"], default="trapezoid", help="Quadrature rule")
    parser.add_argument("--tau-min", type=float, default=0.001, help="Minimum heat trace tau")
    parser.add_argument("--tau-max", type=float, default=1.0, help="Maximum heat trace tau")
    parser.add_argument("--n-tau", type=int, default=100, help="Number of heat trace samples")
    parser.add_argument("--run-otoc", action="store_true", help="Compute OTOC diagnostic")
    parser.add_argument("--otoc-t-max", type=float, default=10.0, help="Maximum OTOC time")
    parser.add_argument("--otoc-n-times", type=int, default=200, help="Number of OTOC time samples")
    return parser.parse_args(argv)


def _json_safe(value: Any) -> Any:
    """Convert NumPy values to JSON-safe objects."""
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
        return {"real": value.real, "imag": value.imag}
    return value


def _setup_matplotlib(out_dir: Path):
    """Import matplotlib with a local cache."""
    plot_cache = out_dir / ".matplotlib"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    return plt


def _balanced_bracket_content(text: str, start: int, open_char: str = "[", close_char: str = "]") -> Optional[str]:
    """Extract balanced bracket content starting at an opening bracket."""
    depth = 0
    body_start = start + 1
    for pos in range(start, len(text)):
        char = text[pos]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[body_start:pos]
    return None


def extract_kernel_expression(text: str) -> str:
    """Extract K(t_i,t_j) expression from SymPy or Wolfram-style text."""
    match = re.search(r"InputForm\s*\[", text)
    if match:
        bracket_pos = text.find("[", match.start())
        content = _balanced_bracket_content(text, bracket_pos)
        if content:
            return content.strip()

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("K(") and "=" in stripped:
            expr = stripped.split("=", 1)[1].strip()
            continuation = []
            for next_line in lines[idx + 1 :]:
                nxt = next_line.strip()
                if not nxt or re.match(r"^[A-Za-z_][A-Za-z0-9_ ()/-]*=", nxt):
                    break
                continuation.append(nxt)
            return " ".join([expr] + continuation).strip()

    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(("Symbolic", "Delta =", "m =", "%")):
            return stripped
    raise ValueError("could not extract kernel expression")


def wolfram_to_sympy_syntax(expr: str) -> str:
    """Minimal Wolfram InputForm to SymPy syntax conversion."""
    converted = expr.strip()
    for old, new in {
        "Cos": "cos",
        "Sin": "sin",
        "Tan": "tan",
        "Log": "log",
        "Exp": "exp",
        "Abs": "Abs",
        "Pi": "pi",
    }.items():
        converted = re.sub(rf"\b{old}\b", new, converted)
    converted = converted.replace("^", "**")
    converted = converted.replace("[", "(").replace("]", ")")
    converted = converted.replace("{", "(").replace("}", ")")
    return converted


def parse_kernel_expression(expr_text: str) -> Tuple[Any, Dict[str, Any]]:
    """Parse symbolic kernel expression into SymPy."""
    if not HAS_SYMPY or sympy is None:
        raise RuntimeError("SymPy is required for V11.4 trace validation")
    ti, tj = sympy.symbols("t_i t_j", real=True)
    Delta = sympy.Symbol("Delta", nonnegative=True)
    m = sympy.Symbol("m", real=True)
    locals_map = {
        "t_i": ti,
        "t_j": tj,
        "ti": ti,
        "tj": tj,
        "Delta": Delta,
        "m": m,
        "I": sympy.I,
        "pi": sympy.pi,
        "Pi": sympy.pi,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "log": sympy.log,
        "exp": sympy.exp,
        "sqrt": sympy.sqrt,
        "Abs": sympy.Abs,
    }
    candidates = [expr_text, wolfram_to_sympy_syntax(expr_text)]
    errors = []
    for candidate in candidates:
        try:
            expr = sympy.sympify(candidate, locals=locals_map)
            return expr, {"used_wolfram_conversion": candidate != expr_text, "parsed": str(expr)}
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError(f"failed to parse kernel expression: {errors}")


def resolve_formula_path(path: Path) -> Path:
    """Resolve common V11.2/V11.2B/V11.2C symbolic formula filename variants."""
    if path.exists():
        return path

    candidates = []
    name = path.name
    replacements = [
        ("_v2_sympy.txt", "_v2B_sympy.txt"),
        ("_v2_sympy.txt", "_v2C_sympy.txt"),
        ("_v2B_sympy.txt", "_v2_sympy.txt"),
        ("_v2B_sympy.txt", "_v2C_sympy.txt"),
        ("_v2C_sympy.txt", "_v2B_sympy.txt"),
        ("_v2C_sympy.txt", "_v2_sympy.txt"),
    ]
    for old, new in replacements:
        if old in name:
            candidates.append(path.with_name(name.replace(old, new)))

    candidates.extend(sorted(path.parent.glob("symbolic_kernel_candidate*_sympy.txt")))
    existing = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            existing.append(candidate)

    if len(existing) == 1:
        print(f"[V11.4][WARN] formula file not found: {path}")
        print(f"[V11.4] using detected formula file: {existing[0]}")
        return existing[0]
    if len(existing) > 1:
        options = "\n  ".join(str(p) for p in existing)
        raise FileNotFoundError(
            f"formula file not found: {path}\n"
            f"Multiple candidate formula files exist. Please pass one explicitly:\n  {options}"
        )

    raise FileNotFoundError(
        f"formula file not found: {path}\n"
        "No symbolic_kernel_candidate*_sympy.txt file was found in the same directory."
    )


def kernel_callable(expr: Any):
    """Construct vectorized callable K(t_i,t_j)."""
    ti, tj = sympy.symbols("t_i t_j", real=True)
    Delta = sympy.Symbol("Delta", nonnegative=True)
    m = sympy.Symbol("m", real=True)
    expr_ts = expr.subs({Delta: sympy.Abs(ti - tj), m: (ti + tj) / 2})
    return sympy.lambdify((ti, tj), expr_ts, modules=["numpy"]), str(expr_ts)


def quadrature_weights(n: int, quad: str) -> np.ndarray:
    """Return quadrature weights on [0,1]."""
    if n < 2:
        raise ValueError("--n-grid must be at least 2")
    if quad != "trapezoid":
        raise ValueError(f"unsupported quadrature rule: {quad}")
    w = np.ones(n, dtype=np.float64) / (n - 1)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def build_integral_operator(fn: Any, n: int, quad: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build weighted self-adjoint integral operator matrix."""
    grid = np.linspace(0.0, 1.0, n, dtype=np.float64)
    ti = grid[:, None]
    tj = grid[None, :]
    try:
        K_grid = np.asarray(fn(ti, tj), dtype=np.complex128)
    except Exception:
        K_grid = np.empty((n, n), dtype=np.complex128)
        for i, x in enumerate(grid):
            K_grid[i, :] = np.asarray(fn(float(x), grid), dtype=np.complex128)
    if K_grid.shape == ():
        K_grid = np.full((n, n), complex(K_grid), dtype=np.complex128)
    if K_grid.shape != (n, n):
        K_grid = np.broadcast_to(K_grid, (n, n)).astype(np.complex128)

    weights = quadrature_weights(n, quad)
    sqrt_w = np.sqrt(weights)
    H = sqrt_w[:, None] * K_grid * sqrt_w[None, :]
    H = 0.5 * (H + H.conj().T)
    return H, K_grid, weights


def health_metrics(H: np.ndarray) -> Dict[str, Any]:
    """Compute finite/self-adjoint/spectral health metrics."""
    finite = bool(np.all(np.isfinite(H)))
    hermitian_error = float(np.linalg.norm(H - H.conj().T) / max(float(np.linalg.norm(H)), EPS)) if H.size else 0.0
    eigvals = np.linalg.eigvalsh(H) if finite else np.asarray([], dtype=float)
    eigvals = np.sort(np.real(eigvals[np.isfinite(eigvals)]))
    if eigvals.size:
        abs_eig = np.abs(eigvals)
        nonzero = abs_eig[abs_eig > EPS]
        condition_proxy = float(abs_eig.max() / max(nonzero.min() if nonzero.size else EPS, EPS))
        return {
            "finite_outputs": finite,
            "hermitian_error": hermitian_error,
            "eig_min": float(eigvals[0]),
            "eig_max": float(eigvals[-1]),
            "eig_range": float(eigvals[-1] - eigvals[0]),
            "eig_std": float(np.std(eigvals)),
            "condition_proxy": condition_proxy,
            "eigvals": eigvals,
        }
    return {
        "finite_outputs": finite,
        "hermitian_error": hermitian_error,
        "eig_min": None,
        "eig_max": None,
        "eig_range": None,
        "eig_std": None,
        "condition_proxy": None,
        "eigvals": eigvals,
    }


def write_eigenvalues(path: Path, eigvals: np.ndarray) -> None:
    """Write eigenvalues CSV."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "eigenvalue"])
        for idx, value in enumerate(eigvals):
            writer.writerow([idx, float(value)])


def safe_heat_trace(eigvals: np.ndarray, taus: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute heat trace with exponent clipping for finite diagnostics."""
    heat = []
    for tau in taus:
        exponent = np.clip(-float(tau) * eigvals, -700.0, 700.0)
        heat.append(float(np.sum(np.exp(exponent))))
    heat_arr = np.asarray(heat, dtype=np.float64)
    return heat_arr, heat_arr / max(eigvals.size, 1)


def write_heat_trace(out_dir: Path, taus: np.ndarray, heat: np.ndarray, heat_norm: np.ndarray) -> None:
    """Write heat trace CSV and plot."""
    with (out_dir / "heat_trace.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "heat_trace", "heat_trace_norm"])
        for row in zip(taus, heat, heat_norm):
            writer.writerow([float(row[0]), float(row[1]), float(row[2])])

    plt = _setup_matplotlib(out_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(taus, heat_norm, label="symbolic integral operator")
    ax.set_xlabel("tau")
    ax.set_ylabel("Tr(exp(-tau H)) / N")
    ax.set_title("V11.4 Heat Trace")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "heat_trace.png", dpi=160)
    plt.close(fig)


def load_operator_eigvals(path: Path) -> np.ndarray:
    """Load learned operator and return Hermitian eigenvalues."""
    H = np.asarray(np.load(path, mmap_mode="r"), dtype=np.complex128)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"learned operator must be square, got {H.shape}")
    H = 0.5 * (H + H.conj().T)
    return np.sort(np.real(np.linalg.eigvalsh(H)))


def compare_spectra(eig: np.ndarray, learned: np.ndarray, taus: np.ndarray, heat: np.ndarray) -> Dict[str, Any]:
    """Compare symbolic integral spectrum with learned operator spectrum."""
    k = min(eig.size, learned.size)
    if k == 0:
        return {"available": False}
    eig_k = eig[:k]
    learned_k = learned[:k]
    spectral_relative_error = float(np.linalg.norm(eig_k - learned_k) / max(float(np.linalg.norm(learned_k)), EPS))
    first_k = min(50, k)
    first_k_error = float(
        np.linalg.norm(eig_k[:first_k] - learned_k[:first_k]) / max(float(np.linalg.norm(learned_k[:first_k])), EPS)
    )
    learned_heat, learned_heat_norm = safe_heat_trace(learned, taus)
    denom = np.maximum(np.abs(learned_heat), EPS)
    heat_rel = np.abs(heat - learned_heat) / denom
    return {
        "available": True,
        "k": int(k),
        "spectral_relative_error": spectral_relative_error,
        "first_k_spectral_error": first_k_error,
        "range_error": float(abs((eig.max() - eig.min()) - (learned.max() - learned.min()))),
        "std_error": float(abs(np.std(eig) - np.std(learned))),
        "heat_trace_relative_error_mean": float(np.mean(heat_rel)),
        "heat_trace_relative_error_max": float(np.max(heat_rel)),
        "learned_heat_trace": learned_heat,
        "learned_heat_trace_norm": learned_heat_norm,
    }


def plot_heat_comparison(out_dir: Path, taus: np.ndarray, heat_norm: np.ndarray, learned_heat_norm: np.ndarray) -> None:
    """Plot heat trace comparison."""
    plt = _setup_matplotlib(out_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(taus, heat_norm, label="symbolic integral")
    ax.plot(taus, learned_heat_norm, label="learned operator", alpha=0.8)
    ax.set_xlabel("tau")
    ax.set_ylabel("normalized heat trace")
    ax.set_title("Heat Trace Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "heat_trace_comparison.png", dpi=160)
    plt.close(fig)


def load_zeta_zeros(path: Path) -> np.ndarray:
    """Load zeta zeros from CSV-like text."""
    vals = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                try:
                    vals.append(float(cell))
                    break
                except ValueError:
                    continue
    arr = np.asarray(vals, dtype=float)
    arr = np.sort(np.abs(arr[np.isfinite(arr)]))
    return arr[arr > 0.0]


def unfolded_spacings(vals: np.ndarray) -> np.ndarray:
    """Return mean-normalized spacings."""
    vals = np.sort(np.asarray(vals, dtype=float))
    if vals.size < 2:
        return np.asarray([], dtype=float)
    spacing = np.diff(vals)
    mean = float(np.mean(spacing))
    return spacing / max(mean, EPS)


def zeta_comparison(eigvals: np.ndarray, zeros: np.ndarray) -> Dict[str, Any]:
    """Compare operator eigenvalue spacings against zeta zero spacings."""
    k = min(eigvals.size, zeros.size)
    if k < 3:
        return {"available": False}
    eig_k = np.sort(eigvals)[:k]
    zeros_k = np.sort(zeros)[:k]
    se = unfolded_spacings(eig_k)
    sz = unfolded_spacings(zeros_k)
    nearest = np.min(np.abs(eig_k[:, None] - zeros_k[None, :]), axis=0)
    return {
        "available": True,
        "k": int(k),
        "zeta_range": float(zeros_k[-1] - zeros_k[0]),
        "operator_range": float(eig_k[-1] - eig_k[0]),
        "spacing_mean_error": float(abs(np.mean(se) - np.mean(sz))),
        "spacing_std_error": float(abs(np.std(se) - np.std(sz))),
        "nearest_spectrum_to_zeta_error": float(np.mean(nearest)),
        "operator_spacings": se,
        "zeta_spacings": sz,
    }


def plot_spacing_comparison(out_dir: Path, operator_spacings: np.ndarray, zeta_spacings: np.ndarray) -> None:
    """Plot spacing histograms for operator and zeta zeros."""
    plt = _setup_matplotlib(out_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(operator_spacings, bins=30, density=True, alpha=0.65, label="operator")
    ax.hist(zeta_spacings, bins=30, density=True, alpha=0.45, label="zeta zeros")
    ax.set_xlabel("normalized spacing")
    ax.set_ylabel("density")
    ax.set_title("Spacing Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "spacing_comparison.png", dpi=160)
    plt.close(fig)


def gue_spacing_diagnostics(eigvals: np.ndarray) -> Dict[str, Any]:
    """Compute GUE/Wigner-surmise spacing diagnostics."""
    spacing = unfolded_spacings(eigvals)
    if spacing.size < 3:
        return {"available": False}
    hist, edges = np.histogram(spacing, bins=40, range=(0.0, min(4.0, max(4.0, float(np.max(spacing))))), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    gue_pdf = (32.0 / (math.pi**2)) * centers**2 * np.exp(-4.0 * centers**2 / math.pi)
    gue_hist_l2 = float(np.sqrt(np.mean((hist - gue_pdf) ** 2)))
    s0 = spacing[:-1]
    s1 = spacing[1:]
    valid = (s0 > EPS) & (s1 > EPS)
    r_vals = np.minimum(s0[valid], s1[valid]) / np.maximum(s0[valid], s1[valid])
    return {
        "available": True,
        "gue_hist_l2": gue_hist_l2,
        "spacing_mean": float(np.mean(spacing)),
        "spacing_std": float(np.std(spacing)),
        "r_stat_mean": float(np.mean(r_vals)) if r_vals.size else None,
        "hist_centers": centers,
        "hist_density": hist,
        "gue_pdf": gue_pdf,
    }


def plot_gue_spacing(out_dir: Path, gue: Dict[str, Any]) -> None:
    """Plot operator spacing histogram vs GUE Wigner surmise."""
    plt = _setup_matplotlib(out_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    centers = np.asarray(gue["hist_centers"], dtype=float)
    hist = np.asarray(gue["hist_density"], dtype=float)
    pdf = np.asarray(gue["gue_pdf"], dtype=float)
    width = centers[1] - centers[0] if centers.size > 1 else 0.1
    ax.bar(centers, hist, width=width, alpha=0.55, label="operator spacing")
    ax.plot(centers, pdf, "r-", label="GUE Wigner surmise")
    ax.set_xlabel("normalized spacing")
    ax.set_ylabel("density")
    ax.set_title("GUE Spacing Diagnostic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gue_spacing_hist.png", dpi=160)
    plt.close(fig)


def compute_otoc(H: np.ndarray, t_max: float, n_times: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute a matrix OTOC diagnostic using spectral time evolution."""
    n = H.shape[0]
    x = np.linspace(0.0, 1.0, n)
    V = np.diag(x)
    W = np.diag(np.cos(2.0 * np.pi * x))
    eigvals, eigvecs = np.linalg.eigh(H)
    W_eig = eigvecs.conj().T @ W @ eigvecs
    V_eig = eigvecs.conj().T @ V @ eigvecs
    times = np.linspace(0.0, float(t_max), int(n_times))
    curve = []
    for time in times:
        phase = np.exp(1j * eigvals * time)
        Wt = phase[:, None] * W_eig * phase.conj()[None, :]
        comm = Wt @ V_eig - V_eig @ Wt
        curve.append(float(np.real(np.trace(comm.conj().T @ comm)) / n))
    C = np.asarray(curve, dtype=float)
    report = {
        "otoc_initial": float(C[0]) if C.size else None,
        "otoc_max": float(np.max(C)) if C.size else None,
        "otoc_final": float(C[-1]) if C.size else None,
        "finite": bool(np.all(np.isfinite(C))),
    }
    return times, C, report


def write_otoc(out_dir: Path, times: np.ndarray, C: np.ndarray, report: Dict[str, Any]) -> None:
    """Write OTOC CSV, plot, and report."""
    with (out_dir / "otoc_curve.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "C"])
        for row in zip(times, C):
            writer.writerow([float(row[0]), float(row[1])])
    plt = _setup_matplotlib(out_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, C)
    ax.set_xlabel("t")
    ax.set_ylabel("C(t)")
    ax.set_title("V11.4 OTOC Diagnostic")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "otoc_curve.png", dpi=160)
    plt.close(fig)
    with (out_dir / "otoc_report.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)


def run_validation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run V11.4 trace/integral-operator validation."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    formula_path = resolve_formula_path(Path(args.formula))
    formula_text = formula_path.read_text(encoding="utf-8")
    expr_text = extract_kernel_expression(formula_text)
    expr, parse_report = parse_kernel_expression(expr_text)
    fn, callable_expr = kernel_callable(expr)

    print(f"[V11.4] building integral operator on n_grid={args.n_grid}")
    H, _, _ = build_integral_operator(fn, int(args.n_grid), str(args.quad))
    finite_outputs = bool(np.all(np.isfinite(H)))
    if not finite_outputs:
        raise FloatingPointError("integral operator contains NaN/Inf entries")
    np.save(out_dir / "integral_operator.npy", H)

    health = health_metrics(H)
    eigvals = health.pop("eigvals")
    write_eigenvalues(out_dir / "eigenvalues.csv", eigvals)

    taus = np.linspace(float(args.tau_min), float(args.tau_max), int(args.n_tau))
    heat, heat_norm = safe_heat_trace(eigvals, taus)
    write_heat_trace(out_dir, taus, heat, heat_norm)

    learned_report: Dict[str, Any] = {"available": False}
    learned_available = False
    if args.learned_operator and Path(args.learned_operator).exists():
        print("[V11.4] comparing learned operator")
        learned_eig = load_operator_eigvals(Path(args.learned_operator))
        learned_report = compare_spectra(eigvals, learned_eig, taus, heat)
        learned_available = bool(learned_report.get("available", False))
        if learned_available:
            plot_heat_comparison(out_dir, taus, heat_norm, np.asarray(learned_report["learned_heat_trace_norm"]))
            learned_report = {k: v for k, v in learned_report.items() if not isinstance(v, np.ndarray)}
        with (out_dir / "learned_comparison.json").open("w", encoding="utf-8") as f:
            json.dump(_json_safe(learned_report), f, indent=2)

    zeta_report: Dict[str, Any] = {"available": False}
    zeta_available = False
    if args.zeta_zeros and Path(args.zeta_zeros).exists():
        print("[V11.4] comparing zeta zeros")
        zeros = load_zeta_zeros(Path(args.zeta_zeros))
        zeta_report = zeta_comparison(eigvals, zeros)
        zeta_available = bool(zeta_report.get("available", False))
        if zeta_available:
            plot_spacing_comparison(out_dir, np.asarray(zeta_report["operator_spacings"]), np.asarray(zeta_report["zeta_spacings"]))
            zeta_report = {k: v for k, v in zeta_report.items() if not isinstance(v, np.ndarray)}
        with (out_dir / "zeta_trace_comparison.json").open("w", encoding="utf-8") as f:
            json.dump(_json_safe(zeta_report), f, indent=2)

    gue_report = gue_spacing_diagnostics(eigvals)
    gue_available = bool(gue_report.get("available", False))
    if gue_available:
        plot_gue_spacing(out_dir, gue_report)
        gue_report_json = {k: v for k, v in gue_report.items() if not isinstance(v, np.ndarray)}
    else:
        gue_report_json = gue_report
    with (out_dir / "gue_spacing_report.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(gue_report_json), f, indent=2)

    otoc_available = False
    if args.run_otoc:
        print("[V11.4] computing OTOC")
        times, C, otoc_report = compute_otoc(H, float(args.otoc_t_max), int(args.otoc_n_times))
        write_otoc(out_dir, times, C, otoc_report)
        otoc_available = bool(otoc_report.get("finite", False))

    report = {
        "version": VERSION,
        "operator_form": OPERATOR_FORM,
        "formula_file": str(formula_path),
        "parse_report": parse_report,
        "callable_expression": callable_expr,
        "n_grid": int(args.n_grid),
        "quad": str(args.quad),
        **health,
        "heat_trace_tau_min": float(args.tau_min),
        "heat_trace_tau_max": float(args.tau_max),
        "n_tau": int(args.n_tau),
        "gue_hist_l2": gue_report_json.get("gue_hist_l2"),
        "r_stat_mean": gue_report_json.get("r_stat_mean"),
        "learned_comparison_available": learned_available,
        "zeta_comparison_available": zeta_available,
        "otoc_available": otoc_available,
        "outputs": {
            "integral_operator": str(out_dir / "integral_operator.npy"),
            "eigenvalues": str(out_dir / "eigenvalues.csv"),
            "heat_trace": str(out_dir / "heat_trace.csv"),
            "heat_trace_plot": str(out_dir / "heat_trace.png"),
            "gue_spacing_report": str(out_dir / "gue_spacing_report.json"),
            "gue_spacing_hist": str(out_dir / "gue_spacing_hist.png") if gue_available else None,
            "learned_comparison": str(out_dir / "learned_comparison.json") if args.learned_operator else None,
            "zeta_trace_comparison": str(out_dir / "zeta_trace_comparison.json") if args.zeta_zeros else None,
            "otoc_curve": str(out_dir / "otoc_curve.csv") if args.run_otoc else None,
        },
        "note": NOTE,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)

    print(VERSION)
    print(f"finite_outputs: {report['finite_outputs']}")
    print(f"hermitian_error: {report['hermitian_error']:.6e}")
    print(f"eig_range: {report['eig_range']}")
    print(f"gue_hist_l2: {report['gue_hist_l2']}")
    print(f"r_stat_mean: {report['r_stat_mean']}")
    print(f"output_dir: {out_dir}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    try:
        run_validation(args)
    except Exception as exc:
        raise RuntimeError(f"V11.4 trace/integral validation failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
