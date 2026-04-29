# V13.2 PDE

## Goal

Use symbolic regression and sparse feature libraries to derive PDE-like interpretations of the learned operator.

## What changed

- Added `analysis/operator_pde_discovery.py` and report generation around PDE-style surrogate models.
- Added feature-library support through `core/pde_feature_library.py`.

## Results

- `runs/operator_pde_report.md` reports normalized error `0.0010713967177783843`.
- The selected equation uses four active terms, including `potential_psi`, `mean_length_psi`, `inv_im_psi`, and `log_im_psi`.
- The report is explicit that the regression is based on reconstructed geometry and a finite feature library.

## Failure modes

- The discovered equation is a surrogate fit, not a validated governing equation.
- The feature library may miss the true structure.
- The coefficients still need held-out validation.

## Conclusion

V13.2 is useful as an interpretability direction, but it does not resolve the operator sensitivity problem and should be treated as exploratory analysis.
# V13.2 PDE

## What was tried
- Sparse, interpretable operator-equation discovery over learned operator artifacts.

## Results
- `runs/operator_pde_report.md` reports a low normalized fit error on the current surrogate task.
- The selected equation is a compact approximation, not an exact recovered law.

## Limitations
- The regression uses reconstructed geometry and a finite feature library.
- Terms still require held-out validation and alternative-kernel checks.
