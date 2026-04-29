# Next Steps

## 1. Fix operator sensitivity

- Make operator outputs respond more strongly to changed Artin-word sets.
- Re-run `make sensitivity` after each operator change.
- Treat this as the highest-priority systems problem.

## 2. Improve executor realism

- Replace more of the current proxy reward with operator-level evaluation.
- Keep bounded reward components, but tie them more directly to meaningful spectral objectives.

## 3. Fix TopologicalLM diversity

- Continue anti-repeat generation work.
- Rebalance the training set away from overrepresented local patterns.
- Optimize for deduplicated performance, not raw repeated samples.

## 4. Integrate RL more tightly

- Use RL and TopologicalLM as complementary candidate generators only after the operator sensitivity problem improves.
- Avoid layering more search logic on top of a weak operator signal.
