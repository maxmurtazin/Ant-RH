# Topological LLM

TopologicalLM is a small next-token model over a braid/DTES DSL.

- `core/dtes_braid_dsl.py` defines the serialized episode format.
- `core/braid_tokenizer.py` tokenizes symbolic braid and numeric tokens.
- `core/topological_llm.py` implements the transformer and constrained generation.
- `core/dtes_braid_executor.py` evaluates generated words with a bounded proxy reward.
- `validation/eval_topological_llm.py` compares random, ACO-best, raw TopologicalLM, and deduplicated variants.
- Current state: valid generation works, but candidate diversity is weak and deduplicated mean reward is worse than random in the latest report.
- This branch is exploratory and currently does not show a reliable gain over simpler baselines.
# Topological LLM

Topological LLM is a next-token model over a braid/DTES DSL. It is an experimental candidate generator, not a validated mathematical solver.

## Role in Ant-RH
- Generates symbolic braid-like candidates.
- Is currently useful only insofar as it outperforms random baselines under the evaluator.
