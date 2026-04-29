# Architecture Overview

Ant-RH is a research pipeline for Hilbert-Polya style operator experiments around zeta-zero structure. The current repository combines symbolic Artin-word generation, operator construction, search loops, evaluation scripts, and Gemma-based analysis agents.

## Main pipeline

1. `core/artin_symbolic_billiard.py` generates Artin words and hyperbolic features.
2. `validation/selberg_trace_loss.py` computes Selberg-style spectral-geometric constraints.
3. `core/artin_operator.py` and `core/artin_operator_structured.py` build candidate operators.
4. `core/artin_aco.py` and `validation/artin_rl_train.py` search the symbolic space with ACO and RL.
5. `validation/operator_stability_report.py` evaluates eigensolver stability and operator quality.
6. `analysis/` agents summarize runs, write journals, build memory, and generate reports.

## Extended pipeline

The TopologicalLM branch adds:

1. `core/dtes_braid_dsl.py` for a braid-like DTES episode representation.
2. `core/braid_tokenizer.py` for symbolic tokenization.
3. `core/topological_llm.py` for transformer-based generation.
4. `core/dtes_braid_executor.py` for proxy validation and reward scoring.
5. `validation/eval_topological_llm.py` for baseline comparison and deduplicated evaluation.

## Honest state

- ACO is not learning reliably. The lab journal marks the ACO loss trend as increasing, and the current `runs/artin_aco_history.csv` shows a near-flat best loss around `44.265`.
- RL reward increased in the logged lab journal run, but that does not fix the ACO plateau.
- Operator sensitivity is weak. `runs/operator_sensitivity_report.json` says: "Artin words barely affect operator. Spectrum insensitive to word changes."
- TopologicalLM can generate valid candidates, but its diversity is poor and deduplicated performance is worse than random in the current report.
- The executor reward is now more stable, but it remains a proxy reward rather than a full operator-level objective.
# Architecture Overview

Ant-RH is a research codebase for Hilbert-Polya style operator experiments around zeta-zero structure.

## Core Pipeline
- Symbolic Artin words and geodesic features are generated first.
- Selberg-style and spectral diagnostics score the symbolic/geometric candidates.
- Operators are built and then used by ACO and RL search loops.
- Stability, PDE-discovery, TopologicalLM, and Gemma analysis stages summarize the results.

## Current State
- Main issue: `scaling_and_alignment`.
- ACO best-loss trend: `increasing`.
- RL mean-reward trend: `increasing`.
- Operator spectral loss: `0.012503465344558741`.
- Operator eigensolver success: `True`.
- TopologicalLM is not yet better than random under the current evaluator.

## Primary Entrypoints
- `make run-v12`: `$(PY) scripts/run_v12_hydra.py --config-name config`
- `make topo-all`: `$(MAKE) topo-train
$(MAKE) topo-eval
$(MAKE) topo-report`
- `make study`: `$(PY) analysis/gemma_project_study.py \
--root . \
--backend llama_cpp \
--llama_cli $(LLAMA_CLI) \
--model_path $(GEMMA_ANALYZER)`
