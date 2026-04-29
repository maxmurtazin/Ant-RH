# Pipeline

## Core loop

ACO / RL -> Artin words -> operator construction -> spectral evaluation -> reward -> logging -> Gemma agents

## Detailed stages

### 1. Symbolic generation

- `make artin` runs `core.artin_symbolic_billiard`.
- Output words represent symbolic PSL(2,Z)-style actions used later by the operator and search stages.

### 2. Constraint and operator stages

- `make selberg` runs `validation.selberg_trace_loss`.
- `make operator` runs `core.artin_operator`.
- These stages convert symbolic inputs into spectral or operator-level diagnostics.

### 3. Search stages

- `make aco` and `make aco-gemma` run ACO search over Artin words.
- `make rl` runs policy optimization in `validation/artin_rl_train.py`.
- Current status: ACO remains unstable or plateaued; RL shows some positive reward movement in the lab journal.

### 4. Stability and analysis

- `make stability` runs operator checks.
- `make analyze-gemma`, `make lab-journal`, `make study`, and `make literature` produce summaries and memory artifacts in `runs/`.

## Extended TopologicalLM branch

TopologicalLM -> braid DSL -> token model -> candidate generation -> DTES executor -> ACO refinement placeholder

- `validation/train_topological_llm.py` builds a mixed ACO/journal/synthetic dataset.
- `core/topological_llm.py` generates token sequences with anti-repeat controls.
- `validation/eval_topological_llm.py` evaluates raw and deduplicated candidates.
- `core/dtes_braid_executor.py` scores validity, length, spectral proxy, stability proxy, and diversity.

## Current results

- `runs/topological_lm/report.md` shows valid generation works (`raw valid_braid_ratio = 0.94`).
- Diversity is weak (`unique_candidate_ratio = 0.25`, `duplicate_count = 150`).
- Deduplicated TopologicalLM mean reward is `-0.7915`, while random is `1.1691`.
- The current conclusion is that TopologicalLM is not better than the random baseline under the current evaluator.
# Pipeline

## Data Flow
1. `artin` generates symbolic words/geodesics.
2. `selberg` computes spectral-geometric consistency losses.
3. `operator` builds the numeric operator from symbolic inputs.
4. `aco` or `aco-gemma` searches symbolic candidates.
5. `rl` trains a policy with operator-based feedback.
6. `stability`, `pde`, `topo-eval`, and analysis agents inspect outcomes.

## Main Outputs
- `runs/artin_*`: symbolic and operator artifacts.
- `runs/artin_rl/`: RL history.
- `runs/operator_stability_report.json`: numerical stability diagnostics.
- `runs/operator_pde_report.md`: PDE-style surrogate discovery report.
- `runs/topological_lm/`: TopologicalLM model, eval, and report files.

## Bottlenecks
- ACO is not learning in the current run.
- Alignment/scaling remains the main operator issue.
- TopologicalLM should be treated as exploratory until it clearly beats random.
