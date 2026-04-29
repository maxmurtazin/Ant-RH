# Ant-RH Gemma Lab Journal

## Lab Entry 2026-04-29T06:14:19+00:00
- Commit: `761b8bed048ff14e128bd7736c6a5ca459212e6a`
- Backend: `rule_based`
- Model: `/Users/machome/models/gemma/gemma-3-1b-it-Q5_K_M.gguf`
- Command/config used: runs/v12_config_used.yaml (if present) + logs/v12/*.log tails
- Key metrics:
  - aco_best_loss_last: 88526.29283751908
  - aco_best_loss_trend: increasing
  - rl_mean_reward_last: 1.365898022428155
  - rl_mean_reward_trend: increasing
  - operator_spectral_loss: 0.012503465344558741
  - operator_spacing_loss: 0.515005380230902
  - operator_eigensolver_succeeded: True
  - selberg_relative_error: 1.0264673145839847
- What worked: RL mean reward increased across updates. Stability check passed; eigensolver succeeded. Spectral loss remains bounded (0.0125035).
- What failed: ACO is not learning in this run. Selberg relative error remains high.
- Interpretation: Learning signal is mixed across modules.
- Next suggested action: Retune ACO exploration parameters (beta, rho) and rerun a 10-iteration smoke test.

## Lab Entry 2026-04-29T06:21:10+00:00
- Commit: `761b8bed048ff14e128bd7736c6a5ca459212e6a`
- Backend: `rule_based`
- Model: `/Users/machome/models/gemma/gemma-3-1b-it-Q5_K_M.gguf`
- Command/config used: runs/v12_config_used.yaml (if present) + logs/v12/*.log tails
- Key metrics:
  - aco_best_loss_last: 88526.29283751908
  - aco_best_loss_trend: increasing
  - rl_mean_reward_last: 1.365898022428155
  - rl_mean_reward_trend: increasing
  - operator_spectral_loss: 0.012503465344558741
  - operator_spacing_loss: 0.515005380230902
  - operator_eigensolver_succeeded: True
  - selberg_relative_error: 1.0264673145839847
- What worked: RL mean reward increased across updates. Stability check passed; eigensolver succeeded. Spectral loss remains bounded (0.0125035).
- What failed: ACO is not learning in this run. Selberg relative error remains high.
- Interpretation: Learning signal is mixed across modules.
- Next suggested action: Retune ACO exploration parameters (beta, rho) and rerun a 10-iteration smoke test.

