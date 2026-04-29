# Current State

## What works

- The end-to-end project structure is in place: symbolic generation, operator construction, ACO, RL, stability, and Gemma-based reporting.
- Operator stability is numerically acceptable in the logged run: eigensolver succeeded and spectral loss was `0.012503465344558741`.
- TopologicalLM can generate valid candidates; the latest raw valid ratio is `0.94`.
- Documentation, project memory, literature memory, and help-agent grounding are all operational.

## What does not work well

- ACO is not learning reliably.
- Operator sensitivity to Artin words is extremely weak.
- TopologicalLM suffers from mode collapse and low diversity.
- Deduplicated TopologicalLM reward is worse than random in the latest report.

## Key metrics

- Current ACO best loss in `runs/artin_aco_history.csv`: about `44.265`.
- Current ACO mean reward in history: roughly `0.13` to `0.17` under rank mode.
- Lab journal RL mean reward last: `1.365898022428155`.
- Selberg relative error in lab journal: `1.0264673145839847`.
- Operator sensitivity diagnosis: "Artin words barely affect operator. Spectrum insensitive to word changes."
- TopologicalLM raw mean reward: `0.9013323550357765`.
- TopologicalLM dedup mean reward: `-0.7915124935111636`.
- Random baseline mean reward: `1.1691048539623852`.
