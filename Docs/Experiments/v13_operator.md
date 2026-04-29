# V13 Operator Improvements

## Goal

Improve operator stability, logging, analysis, and tooling around the baseline search pipeline.

## What changed

- Added structured operator and stabilization paths.
- Added Gemma-based analyzer, journal, project study, literature study, and help tooling.
- Added operator sensitivity diagnostics to test whether Artin words actually influence the operator.

## Results

- The lab journal reports bounded operator spectral loss `0.012503465344558741` and successful eigensolver status.
- `runs/operator_sensitivity_report.json` shows extremely small operator and spectrum movement across different word sets.
- The sensitivity diagnosis is explicit: "Artin words barely affect operator. Spectrum insensitive to word changes."

## Failure modes

- A numerically stable operator is not the same as a sensitive or useful operator.
- Search and planner stages are limited if word changes barely affect the operator.

## Conclusion

V13 improved observability and stability, but also made the main structural weakness visible: operator sensitivity is too low.
