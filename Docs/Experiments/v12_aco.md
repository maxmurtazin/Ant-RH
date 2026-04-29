# V12 ACO Baseline

## Goal

Establish an end-to-end baseline using Artin words, operator construction, ACO, RL, and stability checks.

## What changed

- Added the V12 Hydra-driven pipeline and standard `make` targets around artin, selberg, operator, ACO, RL, and stability.

## Results

- The lab journal reports `aco_best_loss_last = 88526.29283751908` and trend `increasing` for one logged run.
- The current `runs/artin_aco_history.csv` shows best loss hovering around `44.265` with very small variation and `best_reward = 1.0` in rank mode.
- RL mean reward increased in the lab journal run to `1.365898022428155`.

## Failure modes

- ACO is not learning reliably; current history is close to a plateau.
- Selberg relative error remains high.
- Reward shaping improved stability but did not solve the core search problem.

## Conclusion

V12 established the baseline workflow, but the ACO stage remains the weakest part of the current system.
