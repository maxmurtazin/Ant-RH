# Getting started

## Requirements

- Python 3.10+ recommended (uses `from __future__ import annotations` throughout).
- Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Packages: `numpy`, `mpmath`, `matplotlib`, `tqdm` (see [../requirements.txt](../requirements.txt)).

## Working directory

Run scripts from the **repository root** unless a script’s docstring says otherwise. That keeps imports and relative paths (for example `hybrid/`, `validation/`) consistent.

## Full pipeline (recommended)

The bash driver [../run_full_pipeline.sh](../run_full_pipeline.sh) runs, in order:

1. Ground-truth Hardy \(Z\) scan (if the truth JSON is missing).
2. DTES core candidates via `core/fractal_dtes_crosschannel_explore_eta_clean.py`.
3. Colored ants refinement, gap detection, distance analyses, hybrid scan, final distance analysis.

Example:

```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

Override defaults with environment variables (see the `CONFIG` section in the script), for example:

```bash
T_MIN=100 T_MAX=160 N0=2500 ./run_full_pipeline.sh
```

Outputs land under `runs/run_<timestamp>/`.

## Quick smoke test (smaller range)

Use a narrow \([t_{\min}, t_{\max}]\) and moderate `--dps` while testing wiring. High `dps` and fine `--step` on the truth scan are expensive.

## Optional: ETA fractal run + figures

From the repo root:

```bash
python3 run_with_result_figures.py --t-min 10 --t-max 40
```

This uses `fractal_dtes_aco_zeta_eta.py` and `validation/figures_from_results.py` (ensure `validation` is importable; running from root is enough if you invoke `figures_from_results` via the runner).

## Troubleshooting

- **`ModuleNotFoundError`**: Run from repo root, or set `PYTHONPATH` to the repo root.
- **`validate_zeros_and_spacing_eta.py`**: Imports `validate_zeros_and_spacing` as `base`; that module is not shipped in this tree. Use other validation paths (`distance_analysis.py`, truth scan) unless you add the base module yourself.
