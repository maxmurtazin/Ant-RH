# Bottlenecks

## 1. Operator sensitivity

- This is the most serious issue in the current system.
- `runs/operator_sensitivity_report.json` shows tiny operator and spectrum changes across different word sets.
- If Artin words do not move the operator, search cannot produce meaningful gains.

## 2. Reward shaping and realism

- ACO reward shaping is more stable than before, but the search remains near a plateau.
- The TopologicalLM executor reward is bounded and interpretable, but still proxy-based.
- A stable proxy reward is useful for debugging, but it is not a substitute for a stronger operator-level objective.

## 3. TopologicalLM diversity

- Valid generation works, but diversity is poor.
- Latest report: `unique_candidate_ratio = 0.25`, `duplicate_count = 150`.
- Deduplicated mean reward is below the random baseline.
- This indicates mode collapse around a narrow family of candidates.
