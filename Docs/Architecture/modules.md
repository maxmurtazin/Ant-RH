# Modules

## `core/`

- `artin_symbolic_billiard.py`: symbolic Artin-word generation and hyperbolic word construction.
- `artin_operator.py`: baseline operator construction from geodesic inputs.
- `artin_operator_structured.py`: structured operator variant used by some sensitivity and PDE experiments.
- `artin_aco.py`: ant-colony search over symbolic words.
- `artin_rl_env.py`, `artin_rl_policy.py`: RL environment and policy components.
- `spectral_stabilization.py`: stable eigensolver wrapper used by operator and executor code.
- `dtes_braid_dsl.py`: DTES-braid serialization layer.
- `braid_tokenizer.py`: symbolic tokenizer for the TopologicalLM branch.
- `topological_llm.py`: small transformer decoder plus constrained generation.
- `dtes_braid_executor.py`: proxy evaluator for generated braid candidates.
- `gemma_planner.py`: local LLM planner used to propose Artin words.
- `llm_runner.py`: wrapper around `llama-cli`.

## `validation/`

- `selberg_trace_loss.py`: spectral-geometric constraint computation.
- `artin_rl_train.py`: RL training loop.
- `operator_stability_report.py`: eigensolver and stability checks.
- `operator_sensitivity_test.py`: sensitivity diagnostic; current output shows weak dependence on Artin words.
- `train_topological_llm.py`: dataset builder and TopologicalLM trainer.
- `eval_topological_llm.py`: baseline comparison, dedup metrics, and diversity diagnosis.
- `dtes_spectral_validation.py`: spectral validation path for DTES operator experiments.

## `analysis/`

- `gemma_analyzer.py`: metrics-to-summary analyzer.
- `gemma_lab_journal.py`: appends run summaries to the lab journal.
- `gemma_project_study.py`: builds project memory from code and outputs.
- `gemma_literature_study.py`: summarizes literature review files.
- `operator_pde_discovery.py`: sparse PDE-like operator discovery.
- `topological_llm_report.py`: Markdown report synthesis for TopologicalLM results.
- `gemma_docs_builder.py`: generated-docs helper, but the authoritative project state still comes from the underlying run artifacts.

## `help/`

- `gemma_help_agent.py`: interactive assistant grounded in `runs/` data and project memory.
- `local_tts.py`, `local_tts_pyttsx3.py`: local text-to-speech layers.

## Module-level risks

- The codebase has more reporting and helper layers than confirmed algorithmic wins.
- Operator sensitivity remains the main architectural weakness because multiple downstream stages depend on meaningful operator variation.
# Modules

## core/
- `__init__.py`: Core modules for Ant-RH.
- `adaptive_loss_controller.py`: Adaptive loss coefficient controller for DTES geometry learning.
- `artin_aco.py`: Module for pipeline logic or operator/search utilities.
- `artin_billiard.py`: Approximate PSL(2,Z)\\H fundamental domain: |x| <= 1/2 x^2 + y^2 >= 1 y > 0 Returns: points: tensor [N, 2] with columns x,y
- `artin_operator.py`: d(z,w) = arccosh(1 + |z-w|^2 / (2 Im(z) Im(w))). Returns matrix shape (N1, N2).
- `artin_operator_structured.py`: Random smooth values over z (uses x coordinate + a few Gaussian bumps).
- `artin_rl_env.py`: Lightweight symbolic word environment. Exact Selberg/operator eval is handled by training loop; env provides fast proxies.
- `artin_rl_policy.py`: Module for pipeline logic or operator/search utilities.
- `artin_symbolic_billiard.py`: Artin symbolic billiard: PSL(2,Z) words γ = S T^{a1} S T^{a2} ... S T^{ak}, hyperbolic length from trace, ACO-style sampling. Run: python3 -m core.artin_symbolic_billiard --num_samples 10000 --max_length 10 --max_power 7 --out_dir runs/
- `braid_tokenizer.py`: Module for pipeline logic or operator/search utilities.
- `dtes_braid_dsl.py`: Module for pipeline logic or operator/search utilities.
- `dtes_braid_executor.py`: Module for pipeline logic or operator/search utilities.
- `dtes_geometry.py`: Learnable DTES geometry utilities. This module learns a finite embedding whose induced graph Laplacian defines a self-adjoint DTES operator. It is a numerical geometry learning experiment, not a proof method for the Riemann hypothesis.
- `dtes_graph_operator.py`: Graph-DTES operator learning utilities. This module learns a finite self-adjoint DTES graph operator H = L_DTES + V. It is a numerical inverse spectral fitting experiment, not a proof method.
- `dtes_operator_learning.py`: Direct learning of self-adjoint DTES operators. This module is an experimental Hilbert-Polya-inspired numerical fitting tool. It learns a symmetric finite operator whose spectrum is compared with zeta-zero ordinates; it is not a proof method.
- `dtes_operator_physics.py`: Physics-constrained DTES operator learning. This module learns a Schrödinger-type finite operator H = -Delta + V(x) with a fixed local Laplacian and a learnable diagonal potential. It is an experimental spectral fitting diagnostic, not a proof method.
- `dtes_spectral_learning.py`: Experimental DTES spectral-learning utilities. These routines build a symmetric DTES graph operator and compare its spectrum to known zeta-zero ordinates. This is a numerical learning signal, not a proof of the Riemann hypothesis.
- `dtes_spectral_operator.py`: Self-adjoint DTES spectral operator utilities. The operator is a numerical Hilbert-Polya inspired diagnostic built from a DTES pheromone graph and a zeta-derived potential. It is not a proof tool.
- `dtes_trace_tools.py`: Trace-formula diagnostics for experimental DTES spectral operators. These helpers compare finite DTES spectra with zeta-inspired diagnostics. They are numerical research signals for the ACO feedback loop, not RH proof tools.
- `fractal_dtes_crosschannel_explore_eta_clean.py`: fractal_dtes_crosschannel_explore_eta_clean.py Clean standalone DTES-like CrossChannel runner for zeta-zero candidate generation. Why this file exists: - It does NOT import the broken fractal_dtes_aco_zeta_crosschannel.py. - It has built-in: * ETA / live progress * exploration pressure * edge-aware anchors * autosave JSON outputs * distance-analysis-compatible candidate format Core idea: 1. Evaluate Hardy Z / |zeta| on a grid. 2. Build multi-channel DTES score: modulus channel + phase/sign-change channel + multiscale channel + exploration channel. 3. Select candidate centers with coverage pressure so the sampler does not collapse into only a few attractor regions. 4. Refine candidates locally by minimizing |zeta| / bracketing Hardy Z sign changes. 5. Save: - dtes_candidates_explore.json core DTES candidates only - dtes_candidates_explore_edgeaware.json core + boundary anchors - edge_anchors.json anchors only - run_metrics_explore.json run metrics / timings Usage: python3 fractal_dtes_crosschannel_explore_eta_clean.py \ --t_min 100 --t_max 400 \ --n0 2500 \ --ants 100 \ --iters 120 \ --dps 50 \ --output dtes_candidates_explore.json \ --edge_output dtes_candidates_explore_edgeaware.json \ --metrics run_metrics_explore.json Then analyze: python3 distance_analysis.py \ --truth zeros_100_400_precise.json \ --dtes dtes_candidates_explore.json \ --t_min 100 --t_max 400 \ --out distance_explore
- `gemma_planner.py`: Module for pipeline logic or operator/search utilities.
- `gue_losses.py`: Match spacing distribution to approximate GUE Wigner surmise: P(s) = (32/pi^2) s^2 exp(-4s^2/pi)
- `llm_runner.py`: Module for pipeline logic or operator/search utilities.
- `operator_health.py`: Module for pipeline logic or operator/search utilities.
- `pauli.py`: Pauli exclusion helpers for DTES/RL-style state-space constraints.
- `pde_feature_library.py`: Module for pipeline logic or operator/search utilities.
- `spectral_stabilization.py`: Returns: H_stable, report
- `topological_llm.py`: Module for pipeline logic or operator/search utilities.

## analysis/
- `gemma_analyzer.py`: Analysis or report-generation utility.
- `gemma_docs_builder.py`: (.*?)
- `gemma_lab_journal.py`: Analysis or report-generation utility.
- `gemma_literature_study.py`: Analysis or report-generation utility.
- `gemma_paper_writer.py`: Analysis or report-generation utility.
- `gemma_project_study.py`: Analysis or report-generation utility.
- `operator_formula_report.py`: Analysis or report-generation utility.
- `operator_pde_discovery.py`: Analysis or report-generation utility.
- `operator_pde_report.py`: Analysis or report-generation utility.
- `operator_symbolic_regression.py`: Analysis or report-generation utility.
- `topological_llm_report.py`: Analysis or report-generation utility.

## Notes
- Module descriptions are generated from lightweight summaries, not full source reprints.
- Use the source files for implementation details and exact APIs.
