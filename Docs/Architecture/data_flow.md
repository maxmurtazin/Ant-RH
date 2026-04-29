# Data Flow

## Core pipeline data flow

1. Symbolic words are produced by the Artin-word layer.
2. Word-derived features are passed into Selberg-style loss computation and operator builders.
3. Search loops produce updated candidate words, rewards, and history rows.
4. Results are written to `runs/` as CSV, JSON, and Markdown artifacts.
5. Gemma agents read those artifacts and generate summaries, journals, memory files, and help responses.

## TopologicalLM data flow

logs -> DSL -> tokenizer -> model -> candidates -> executor -> reward

### Logs and artifacts

- `runs/artin_aco_history.csv`
- `runs/artin_aco_best.json`
- `runs/lab_journal.jsonl` if present

### Serialization

- `core/dtes_braid_dsl.py` converts episodes into a structured text form.

### Tokenization and modeling

- `core/braid_tokenizer.py` maps DSL tokens into ids.
- `validation/train_topological_llm.py` builds the dataset and trains the model.

### Candidate generation

- `core/topological_llm.py` samples token sequences with temperature, top-k, top-p, repetition penalty, and no-repeat n-gram controls.

### Evaluation

- `validation/eval_topological_llm.py` decodes words, deduplicates candidates, and compares raw vs deduplicated outcomes.
- `core/dtes_braid_executor.py` assigns a bounded proxy reward using validity, length, spectral proxy, stability proxy, and diversity terms.

## Current weak points in the data flow

- Operator sensitivity is low, so changed words often produce nearly identical operator behavior.
- The executor is still proxy-based, so high reward does not yet imply meaningful operator improvement.
- TopologicalLM currently collapses around a narrow family of candidates near `[4, -2, -1, ...]`.
