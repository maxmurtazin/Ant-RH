# Selberg-Style Constraints

Selberg-style constraints are used here as spectral-geometric consistency checks.

- The main implementation path is `validation/selberg_trace_loss.py`.
- The goal is to compare symbolic/geometric information against target spectral behavior.
- These constraints contribute diagnostic signal for operator-search experiments.
- The lab journal reports `selberg_relative_error = 1.0264673145839847`, which is still high.
- This means the current symbolic-to-spectral alignment is not yet strong.
- In this repository, Selberg-style losses are practical evaluation terms, not proofs.
