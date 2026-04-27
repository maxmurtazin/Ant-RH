from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import sympy

    HAS_SYMPY = True
except Exception:
    sympy = None
    HAS_SYMPY = False


LABEL = "Symbolic locality diagnostic — not proof of RH"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json_safe(value: Any) -> Any:
    """Convert NumPy/SymPy values to JSON-safe objects."""
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse V11.3 locality detector CLI arguments."""
    parser = argparse.ArgumentParser(description="Ant-RH V11.3 symbolic locality/nonlocality detector.")
    parser.add_argument("--formula", required=True, help="Path to symbolic kernel formula text")
    parser.add_argument("--operator", default=None, help="Optional learned operator .npy used for provenance")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--series-order", type=int, default=4, help="Local series order around Delta=0")
    parser.add_argument("--tail-grid", type=int, default=200, help="Number of Delta samples for numerical tail profile")
    parser.add_argument("--wolfram", action="store_true", help="Run optional Wolfram Engine locality checks")
    parser.add_argument("--wolfram-timeout", type=float, default=30.0, help="Wolfram TimeConstrained timeout")
    parser.add_argument("--force-pde", action="store_true", help="Emit PDE candidate even when classified nonlocal")
    return parser.parse_args(argv)


def _balanced_bracket_content(text: str, start: int, open_char: str = "[", close_char: str = "]") -> Optional[str]:
    """Extract balanced bracket content starting at an opening bracket index."""
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
    """Extract the kernel expression from SymPy or Wolfram-style formula text."""
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
                next_stripped = next_line.strip()
                if not next_stripped or re.match(r"^[A-Za-z_][A-Za-z0-9_ ]*=", next_stripped):
                    break
                continuation.append(next_stripped)
            return " ".join([expr] + continuation).strip()

    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(("Symbolic", "Delta =", "m =", "%")):
            return stripped
    raise ValueError("could not extract kernel expression from formula text")


def wolfram_to_sympy_syntax(expr: str) -> str:
    """Apply minimal Wolfram InputForm to SymPy syntax conversion."""
    converted = expr.strip()
    replacements = {
        "Cos": "cos",
        "Sin": "sin",
        "Tan": "tan",
        "Log": "log",
        "Exp": "exp",
        "Abs": "Abs",
        "Pi": "pi",
        "E^": "exp",
    }
    for old, new in replacements.items():
        converted = re.sub(rf"\b{old}\b", new, converted)
    converted = converted.replace("^", "**")
    converted = converted.replace("[", "(").replace("]", ")")
    converted = converted.replace("{", "(").replace("}", ")")
    return converted


def parse_kernel_expression(expr_text: str) -> Tuple[Any, Dict[str, Any]]:
    """Parse a kernel expression into SymPy."""
    if not HAS_SYMPY or sympy is None:
        raise RuntimeError("SymPy is required for locality detection")

    ti, tj = sympy.symbols("t_i t_j", real=True)
    Delta = sympy.Symbol("Delta", nonnegative=True)
    m = sympy.Symbol("m", real=True)
    s = sympy.Symbol("s", real=True)
    t = sympy.Symbol("t", real=True)

    locals_map = {
        "t_i": ti,
        "t_j": tj,
        "ti": ti,
        "tj": tj,
        "Delta": Delta,
        "m": m,
        "s": s,
        "t": t,
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
            return expr, {"input": expr_text, "parsed": str(expr), "used_wolfram_conversion": candidate != expr_text}
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError(f"failed to parse formula as SymPy/Wolfram expression: {errors}")


def local_kernel_expression(expr: Any) -> Tuple[Any, Any, Any]:
    """Substitute Delta -> s, m -> t, t_i -> t+s/2, t_j -> t-s/2."""
    ti, tj = sympy.symbols("t_i t_j", real=True)
    Delta = sympy.Symbol("Delta", nonnegative=True)
    m = sympy.Symbol("m", real=True)
    s = sympy.Symbol("s", real=True)
    t = sympy.Symbol("t", real=True)
    return expr.subs({Delta: s, m: t, ti: t + s / 2, tj: t - s / 2}), s, t


def compute_local_series(expr: Any, order: int) -> Tuple[Any, str, str]:
    """Compute local series around the diagonal."""
    K_local, s, _ = local_kernel_expression(expr)
    try:
        series = sympy.series(K_local, s, 0, order).removeO()
    except Exception:
        series = sympy.series(K_local.xreplace({sympy.Abs(s): s}), s, 0, order).removeO()
    return series, str(series), sympy.latex(series)


def symbolic_nonlocal_score(expr: Any) -> Tuple[float, List[str]]:
    """Score symbolic nonlocal signatures."""
    Delta = sympy.Symbol("Delta", nonnegative=True)
    score = 0.0
    reasons: List[str] = []

    if expr.has(sympy.log):
        score += 2.0
        reasons.append("log_terms")

    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.Pow):
            base, exponent = node.args
            try:
                if base.has(Delta) and exponent.is_number and float(exponent) < 0:
                    score += 3.0
                    reasons.append("inverse_delta_power")
            except Exception:
                continue

    if expr.has(sympy.sin) or expr.has(sympy.cos):
        score += 1.0
        reasons.append("oscillatory_terms")

    for node in sympy.preorder_traversal(expr):
        if (node.func in (sympy.sin, sympy.cos)) and any(arg.has(sympy.log) for arg in node.args):
            score += 3.0
            reasons.append("log_periodic_oscillation")

    expr_str = str(expr)
    if "t_i*log" in expr_str or "t_j*log" in expr_str:
        score += 4.0
        reasons.append("cross_log_coupling")

    if "1/(Delta + 1)" in expr_str or "log(Delta + 1)" in expr_str:
        reasons.append("weak_delta_tail")

    return float(score), sorted(set(reasons))


def evaluate_tail_profile(expr: Any, tail_grid: int) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluate K(Delta,m=0.5) and compute tail diagnostics."""
    Delta = sympy.Symbol("Delta", nonnegative=True)
    m = sympy.Symbol("m", real=True)
    ti, tj = sympy.symbols("t_i t_j", real=True)
    kernel = expr.subs({m: 0.5, ti: 0.5 + Delta / 2, tj: 0.5 - Delta / 2})
    fn = sympy.lambdify(Delta, kernel, modules=["numpy"])
    grid = np.linspace(1e-6, 1.0, max(10, int(tail_grid)))
    try:
        values = np.asarray(fn(grid), dtype=np.complex128)
    except Exception:
        values = np.array([complex(fn(float(x))) for x in grid], dtype=np.complex128)
    abs_values = np.abs(values)
    finite = np.isfinite(abs_values)
    grid = grid[finite]
    abs_values = abs_values[finite]
    if grid.size == 0:
        raise FloatingPointError("tail profile contains no finite values")

    near_mask = grid < 0.05
    tail_mask = grid > 0.5
    near_mean = float(np.mean(abs_values[near_mask])) if np.any(near_mask) else float(np.mean(abs_values[:1]))
    tail_mean = float(np.mean(abs_values[tail_mask])) if np.any(tail_mask) else float(np.mean(abs_values[-1:]))
    tail_ratio = float(tail_mean / max(near_mean, 1e-12))

    positive = abs_values > 1e-14
    if np.count_nonzero(positive) >= 3:
        beta = float(np.polyfit(np.log1p(grid[positive]), np.log(abs_values[positive]), 1)[0])
        tail_decay_slope = -beta
    else:
        tail_decay_slope = None

    profile = np.column_stack([grid, abs_values])
    metrics = {
        "tail_ratio": tail_ratio,
        "tail_decay_slope": tail_decay_slope,
        "near_mean_abs": near_mean,
        "tail_mean_abs": tail_mean,
        "tail_grid": int(grid.size),
    }
    return metrics, profile


def classify_locality(score: float, tail_ratio: float) -> str:
    """Map symbolic and numeric diagnostics to a locality class."""
    if score >= 6.0 or tail_ratio > 0.1:
        return "strongly_nonlocal"
    if score >= 3.0 or tail_ratio > 0.03:
        return "weakly_nonlocal"
    return "approximately_local"


def write_tail_profile(out_dir: Path, profile: np.ndarray) -> Optional[str]:
    """Write tail profile CSV and PNG if matplotlib is available."""
    csv_path = out_dir / "tail_profile.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Delta", "abs_K"])
        for delta, value in profile:
            writer.writerow([float(delta), float(value)])

    png_path = out_dir / "tail_profile.png"
    plot_cache = out_dir / ".matplotlib"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(profile[:, 0], profile[:, 1])
        ax.set_xlabel("Delta")
        ax.set_ylabel("|K(Delta, m=0.5)|")
        ax.set_title("V11.3 Kernel Tail Profile")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        return str(png_path)
    except Exception as exc:
        (out_dir / "tail_profile_plot_error.txt").write_text(str(exc), encoding="utf-8")
        return None


def write_operator_form(
    out_dir: Path,
    locality_class: str,
    local_series: Any,
    force_pde: bool,
) -> Dict[str, Optional[str]]:
    """Write PDE approximation or nonlocal integral operator form."""
    s = sympy.Symbol("s", real=True)
    t = sympy.Symbol("t", real=True)
    outputs: Dict[str, Optional[str]] = {"pde_candidate": None, "nonlocal_integral_operator": None}

    if locality_class == "approximately_local" or force_pde:
        expanded = sympy.expand(local_series)
        c0 = sympy.simplify(expanded.coeff(s, 0))
        c1 = sympy.simplify(expanded.coeff(s, 1))
        c2 = sympy.simplify(expanded.coeff(s, 2))
        text = (
            f"{LABEL}\n\n"
            "Local PDE candidate from diagonal expansion:\n"
            f"a0(t) = {c0}\n"
            f"a1(t) = {c1}\n"
            f"a2(t) = {c2}\n\n"
            "(Hf)(t) ≈ a0(t) f(t) + a2(t) d^2 f/dt^2\n"
            "Odd coefficient a1(t) is reported as a symmetry diagnostic.\n"
        )
        path = out_dir / "pde_candidate.txt"
        path.write_text(text, encoding="utf-8")
        (out_dir / "pde_candidate_latex.tex").write_text(
            "% " + LABEL + "\n"
            + "\\[\n"
            + f"a_0(t) = {sympy.latex(c0)},\\quad a_1(t) = {sympy.latex(c1)},\\quad a_2(t) = {sympy.latex(c2)}\n"
            + "\\]\n",
            encoding="utf-8",
        )
        outputs["pde_candidate"] = str(path)
    else:
        path = out_dir / "nonlocal_integral_operator.txt"
        path.write_text(
            f"{LABEL}\n\n"
            "The kernel is classified as nonlocal. Use the integral operator form:\n\n"
            "(Hf)(t) = integral K(t,s) f(s) ds\n\n"
            "This is an operator diagnostic, not a proof of RH.\n",
            encoding="utf-8",
        )
        outputs["nonlocal_integral_operator"] = str(path)
    return outputs


def _sympy_to_wolfram(expr: Any) -> str:
    """Convert a SymPy expression string to simple Wolfram-like syntax."""
    text = str(expr)
    replacements = {
        "sin": "Sin",
        "cos": "Cos",
        "log": "Log",
        "exp": "Exp",
        "Abs": "Abs",
        "pi": "Pi",
        "**": "^",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def run_wolfram_checks(expr: Any, out_dir: Path, order: int, timeout: float) -> Dict[str, Any]:
    """Run optional Wolfram Engine locality checks safely."""
    executable = shutil.which("wolframscript")
    report: Dict[str, Any] = {"available": executable is not None}
    if executable is None:
        return report

    wolfram_expr = _sympy_to_wolfram(expr)
    code = (
        f"expr = {wolfram_expr}; "
        f"series = TimeConstrained[FullSimplify[FunctionExpand[Series[expr /. Delta -> s, {{s, 0, {int(order)}}}]]], {float(timeout)}, $Failed]; "
        f"tail = TimeConstrained[FullSimplify[FunctionExpand[Limit[expr, Delta -> Infinity]]], {float(timeout)}, $Failed]; "
        "Print[\"SERIES_START\"]; Print[InputForm[series]]; "
        "Print[\"TAIL_START\"]; Print[InputForm[tail]];"
    )
    try:
        proc = subprocess.run(
            [executable, "-code", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )
    except Exception as exc:
        report.update({"ok": False, "error": str(exc)})
        return report

    output = proc.stdout
    report.update({"ok": proc.returncode == 0, "returncode": proc.returncode, "stderr": proc.stderr.strip()})
    series_text = ""
    tail_text = ""
    if "SERIES_START" in output and "TAIL_START" in output:
        series_text = output.split("SERIES_START", 1)[1].split("TAIL_START", 1)[0].strip()
        tail_text = output.split("TAIL_START", 1)[1].strip()
    (out_dir / "wolfram_local_series.txt").write_text(series_text, encoding="utf-8")
    (out_dir / "wolfram_tail_limit.txt").write_text(tail_text, encoding="utf-8")
    report.update({"series_file": "wolfram_local_series.txt", "tail_file": "wolfram_tail_limit.txt"})
    with (out_dir / "wolfram_locality_report.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)
    return report


def run_detector(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the V11.3 locality detector."""
    if not HAS_SYMPY or sympy is None:
        raise RuntimeError("SymPy is required for V11.3 locality detection")
    formula_path = Path(args.formula)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = formula_path.read_text(encoding="utf-8")
    expr_text = extract_kernel_expression(text)
    expr, parse_report = parse_kernel_expression(expr_text)

    local_series, local_series_text, local_series_latex = compute_local_series(expr, int(args.series_order))
    (out_dir / "local_series.txt").write_text(f"{LABEL}\n\n{local_series_text}\n", encoding="utf-8")
    (out_dir / "local_series_latex.tex").write_text("% " + LABEL + "\n" + local_series_latex + "\n", encoding="utf-8")

    score, reasons = symbolic_nonlocal_score(expr)
    tail_metrics, profile = evaluate_tail_profile(expr, int(args.tail_grid))
    with (out_dir / "tail_test.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(tail_metrics), f, indent=2)
    tail_plot = write_tail_profile(out_dir, profile)

    locality_class = classify_locality(score, float(tail_metrics["tail_ratio"]))
    operator_outputs = write_operator_form(out_dir, locality_class, local_series, bool(args.force_pde))
    wolfram_report = run_wolfram_checks(expr, out_dir, int(args.series_order), float(args.wolfram_timeout)) if args.wolfram else {"available": False, "requested": False}

    report: Dict[str, Any] = {
        "diagnostic_label": LABEL,
        "formula": str(formula_path),
        "operator": str(args.operator) if args.operator else None,
        "locality_class": locality_class,
        "symbolic_nonlocal_score": score,
        "reasons": reasons,
        "tail_ratio": tail_metrics["tail_ratio"],
        "tail_decay_slope": tail_metrics["tail_decay_slope"],
        "tail_metrics": tail_metrics,
        "local_series": local_series_text,
        "parse_report": parse_report,
        "outputs": {
            "locality_report": str(out_dir / "locality_report.json"),
            "local_series": str(out_dir / "local_series.txt"),
            "local_series_latex": str(out_dir / "local_series_latex.tex"),
            "tail_test": str(out_dir / "tail_test.json"),
            "tail_profile_csv": str(out_dir / "tail_profile.csv"),
            "tail_profile_png": tail_plot,
            **operator_outputs,
        },
        "wolfram_report": wolfram_report,
        "note": "Symbolic locality/nonlocality diagnostic for operator research; not a proof of RH.",
    }
    with (out_dir / "locality_report.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)

    print(LABEL)
    print(f"locality_class: {locality_class}")
    print(f"symbolic_nonlocal_score: {score:.3f}")
    print(f"reasons: {', '.join(reasons) if reasons else 'none'}")
    print(f"tail_ratio: {tail_metrics['tail_ratio']:.6g}")
    print(f"tail_decay_slope: {tail_metrics['tail_decay_slope']}")
    print(f"output_dir: {out_dir}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    try:
        run_detector(args)
    except Exception as exc:
        raise RuntimeError(f"V11.3 locality detector failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
