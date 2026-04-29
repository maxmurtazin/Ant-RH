# Debugging

## `llama-cli` hanging or failing

- Run `make gemma-test` to confirm the binary is available.
- The project previously needed fixes for interactive `llama-cli` hangs in the help agent; if hangs return, reduce concurrency and check subprocess timeouts.
- If `make help-chat` stalls after one turn, inspect `help/gemma_help_agent.py` and `core/llm_runner.py`.

## Hydra config errors

- `make run-v12`, `make smoke-v12`, and `make full-v12` all depend on valid Hydra configs.
- If a run fails early, check `configs/` and the `scripts/run_v12_hydra.py` entrypoint.
- Prefer `make smoke-v12` before a longer run when configs have changed.

## Context overflow in Gemma agents

- Project-study and literature-study agents use prompt truncation because local Gemma has a limited context window.
- If summaries degrade or fail, reduce prompt size rather than adding more raw text.

## ACO plateau

- Current issue: ACO is not learning reliably.
- The lab journal says the best-loss trend is increasing.
- `runs/artin_aco_history.csv` shows a near-flat best loss around `44.265`, which is consistent with plateau behavior.
- Retune exploration parameters before interpreting small reward changes as real progress.

## Duplicate candidates in TopologicalLM

- Current issue: generation collapses around a narrow pattern family.
- The latest TopologicalLM report shows `unique_candidate_ratio = 0.25` and `duplicate_count = 150`.
- Use the current anti-repeat controls in `core/topological_llm.py`, but do not assume they fully solve the problem.
- Always compare raw and deduplicated metrics in `runs/topological_lm/report.md`.

## Operator sensitivity

- `runs/operator_sensitivity_report.json` is currently the clearest failure signal.
- The report says Artin words barely affect the operator.
- Until this improves, downstream search and TopologicalLM gains should be treated cautiously.
# Debugging

## Common issues
- If `llama-cli` is missing, run `make gemma-test` and install `llama.cpp`.
- If ACO quality is poor, accept that current runs show non-learning and retune exploration before overinterpreting results.
- If operator metrics drift, inspect `runs/operator_stability_report.json` and rerun `make stability`.
- If TopologicalLM does not beat random, treat it as exploratory and inspect `make topo-eval` outputs before claiming progress.

## Useful commands
```bash
make aco-gemma
make stability
make topo-eval
make study
```
