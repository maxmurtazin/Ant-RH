# Ant-RH Documentation

Ant-RH is a research codebase for zeta-zero and Hilbert-Polya style operator experiments. The repository combines symbolic Artin-word generation, Selberg-style constraints, operator construction, ACO/RL search, stability checks, and Gemma-based reporting.

## Current status

- ACO remains unstable or plateaued.
- Operator sensitivity to Artin words is weak in the current implementation.
- TopologicalLM can generate valid candidates, but mode collapse is present.
- The current TopologicalLM report shows deduplicated mean reward below the random baseline.

## Read first

- `Architecture/overview.md`
- `Architecture/pipeline.md`
- `Architecture/data_flow.md`
- `Reports/current_state.md`

## Sections

- `Architecture/`: system layout, pipeline, modules, and data flow.
- `Concepts/`: DTES, Artin billiard, Selberg-style constraints, operator search, and TopologicalLM.
- `Experiments/`: grounded summaries of V12, V13, V13.2 PDE, and VNext TopologicalLM.
- `Guides/`: quickstart, command reference, and debugging notes.
- `Reports/`: current state, bottlenecks, and prioritized next steps.

## Core commands

```bash
make run-v12
make analyze-gemma
make topo-all
make help-chat
```
