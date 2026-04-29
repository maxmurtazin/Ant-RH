# Operator Search

Operator search is the task of building candidate operators whose spectral behavior better matches the target constraints.

- `core/artin_operator.py` and `core/artin_operator_structured.py` are the main construction paths.
- Search is driven by ACO, RL, and auxiliary diagnostics.
- `validation/operator_stability_report.py` checks whether the numerical operator remains usable.
- `validation/operator_sensitivity_test.py` checks whether changing Artin words materially changes the operator.
- Current result: sensitivity is very low, with diagnosis "Artin words barely affect operator. Spectrum insensitive to word changes."
- This is a core bottleneck because search cannot improve much if the operator barely changes.
# Operator Search

Operator search is the process of building and adjusting candidate operators whose spectra are compared against zeta-zero-derived targets or proxy diagnostics.

## Role in Ant-RH
- Uses symbolic inputs, spectral losses, and stability checks.
- Feeds ACO, RL, and PDE-style analysis stages.
