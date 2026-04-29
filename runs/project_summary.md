# Project Summary

Ant-RH combines Artin symbolic generation, Selberg-style validation, operator construction, ACO/RL search, stability analysis, and Gemma-assisted reporting/help tooling.

## Main Modules
- `core/`: operator construction, ACO, RL environment/policy, stabilization, PDE feature library.
- `validation/`: training/validation scripts, Selberg loss, stability reports, experiment runners.
- `analysis/`: Gemma summaries, journal/paper/PDE studies.
- `help/`: local Gemma help agent and TTS layers.
- `configs/`: Hydra V12 pipeline configs.

## Pipeline Stages
- artin -> selberg -> operator -> aco -> rl -> stability -> analysis/reporting.

## Bottlenecks
- ACO learning remains weak in current runs.
- Operator alignment/scaling remains a recurring issue.
- Reporting/help stack is growing faster than consolidated project memory.

## Next Best Actions
- Run `make study` after each major change to refresh project memory.
- Keep `make analyze-gemma`, `make lab-journal`, and `make pde` in the post-run loop.
- Use the refreshed memory to tighten help-agent answers and pipeline triage.
