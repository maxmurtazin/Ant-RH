# Ant-RH documentation

Extended documentation for the **Ant-RH** repository: DTES-guided exploration of Riemann zeta zeros on the critical line \(\mathrm{Re}(s)=\tfrac12\).

## Contents

| Document | Description |
|----------|-------------|
| [Getting started](getting-started.md) | Dependencies, layout, running the full pipeline |
| [Repository layout](repository-layout.md) | Folders, main modules, which script to use when |
| [CLI reference](cli-reference.md) | Command-line flags for runners and tools |
| [Data formats](data-formats.md) | JSON shapes for truth, candidates, hybrid output |
| [Pipeline and algorithms](pipeline-and-algorithms.md) | End-to-end flow and relation to the math in `README.md` |

## Root README

The project abstract, formulas, hybrid recovery sketch, and complexity notes live in the repository root: [../README.md](../README.md).

## DTES Spectral Operator Validation

The repository includes an experimental DTES spectral diagnostic that maps an ACO pheromone graph into a symmetric graph operator:

- `core/dtes_spectral_operator.py` builds a self-adjoint operator `H = L + diag(V)` from a symmetrized pheromone matrix and a zeta-derived potential.
- `validation/dtes_spectral_validation.py` compares the unfolded operator spectrum with unfolded Riemann zeta zeros and reports global spectral alignment metrics.
- ETA ACO runs export `spectral_ready_result.json` with `t_grid`, `zeta_abs`, `pheromone_matrix`, `true_zeros`, and `spectral_ready=true`.

Run it directly after a spectral-ready ACO export:

```bash
python3 validation/dtes_spectral_validation.py \
  --input fractal_dtes_aco_eta_output/spectral_ready_result.json \
  --out fractal_dtes_aco_eta_output/spectral_report.json \
  --k 50 \
  --potential-mode neglog
```

Or enable it in the ETA figure runner:

```bash
python3 run_with_result_figures.py --spectral-validation --spectral-k 50
```

Expected outputs include `spectral_report.json`, `spectral_ready_result.json`, `spectral_eigenvalues.json`, a CSV of eigenvalues from the validation CLI, and optional plots when `matplotlib` is available.

This is a numerical Hilbert-Polya inspired diagnostic, not a proof of the Riemann Hypothesis. Keep local zero recovery metrics separate from global spectral alignment metrics.
