# Topological LLM Report

## 1. Goal
Train a small topological next-token model over DTES-Braid episodes to heuristically generate braid/operator candidates.

## 2. Dataset
Dataset mixes ACO-derived DTES-Braid episodes, optional lab journal signals, and synthetic fallback episodes.

## 3. Training
- Final train loss: `0.7905545830726624`
- Final perplexity: `2.2046186923980713`
- Epochs logged: `20`

## 4. Candidate Validity
TopologicalLM raw:
- valid_braid_ratio: `0.87`
- rejection_rate: `0.13`
- reward_mean: `-0.09235573202134302`
- reward_median: `1.1966154546956973`
- reward_std: `3.8407169784179613`
- unique_candidate_ratio: `0.30959752321981426`
- duplicate_count: `223`

TopologicalLM dedup:
- valid_braid_ratio: `0.87`
- rejection_rate: `0.13`
- reward_mean: `-0.09235573202134302`
- reward_median: `1.1966154546956973`
- reward_std: `3.8407169784179613`

## 5. Baseline Comparison
- `random`: best_reward=2.177827279755306 mean_reward=1.2478195405035564 median=1.2042420195288748 std=1.1665329630864845 valid_ratio=0.99
- `ACO-only existing best`: best_reward=1.820554700558875 mean_reward=1.820554700558875 median=1.820554700558875 std=0.0 valid_ratio=1.0
- `TopologicalLM-only raw`: best_reward=2.177827279755306 mean_reward=-0.09235573202134302 median=1.1966154546956973 std=3.8407169784179613 valid_ratio=0.87
- `TopologicalLM-only dedup`: best_reward=2.177827279755306 mean_reward=-0.09235573202134302 median=1.1966154546956973 std=3.8407169784179613 valid_ratio=0.87
- `TopologicalLM + ACO raw`: best_reward=2.177827279755306 mean_reward=-0.07152887809859182 median=1.1966154546956973 std=3.848235707926686 valid_ratio=0.87
- `TopologicalLM + ACO dedup`: best_reward=2.177827279755306 mean_reward=-0.07152887809859182 median=1.1966154546956973 std=3.848235707926686 valid_ratio=0.87

## 6. Reward Diagnosis
- advantage_over_random: `-1.3381614249846687`
- Diagnosis: TopologicalLM shows a measurable mean-reward difference versus random. Candidate diversity is low; increase temperature/top_k or add anti-repeat penalty. Candidate diversity remains low.
- Report this honestly: if the mean-reward advantage is near zero, the model is not yet better than random under the current executor.

## 7. Component Trends
- `random` component means: validity=1.4748743718592965, length=-1.5641905456723433, spectral=-0.10385304233642034, stability=0.8894472361809045, diversity=0.7090212969609955
- `ACO-only existing best` component means: validity=1.5, length=-0.5187881749403112, spectral=-0.003384545304302781, stability=1.0, diversity=0.4166666666666667
- `TopologicalLM-only` component means: validity=1.175, length=-1.6005350536258975, spectral=-1.3029445544147435, stability=-0.43, diversity=0.704281746031746
- `TopologicalLM + ACO refinement placeholder` component means: validity=1.175, length=-1.5757686473676968, spectral=-1.3029445544147435, stability=-0.43, diversity=0.7465

## 8. Top Candidates
- word=[4, -2, -1] reward=2.177827279755306 valid=True spectral_error=0.0033902793449763907
- word=[-1, -7, -4, -2] reward=2.0568707104385426 valid=True spectral_error=0.0033902793449763907
- word=[-1, 4, -2, -1, -1] reward=1.9935075373054003 valid=True spectral_error=0.0033902793449763907
- word=[4, -2, -1, -5, 1] reward=1.9747977171062912 valid=True spectral_error=0.0033902793449763907
- word=[4, -2, -1, -1, 1, -8] reward=1.9525215955935875 valid=True spectral_error=0.003390279344976354
- word=[4, -2, -1, -1, 5] reward=1.9481365062332876 valid=True spectral_error=0.0033902793449763907
- word=[5, 4, -2, -1, -1] reward=1.9481365062332876 valid=True spectral_error=0.0033902793449763907
- word=[4, 4, -2, -1, -1] reward=1.9435075373054003 valid=True spectral_error=0.0033902793449763907
- word=[1, -5, -3] reward=1.9243729784845458 valid=True spectral_error=0.0033902793449763907
- word=[-2, 4, -2, -1, -1] reward=1.9081365062332876 valid=True spectral_error=0.0033902793449763907

## 9. Limitations
This is a heuristic generator and does not prove RH.
The MVP uses symbolic token bins and proxy executor rewards; it is not a mathematically validated proof engine.
If TopologicalLM and random remain close under bounded rewards, the current model or dataset is not yet carrying useful structure.

## 10. Next steps
- Add real DTES-braid trajectories instead of synthetic fallbacks.
- Replace proxy executor terms with full structured operator evaluation.
- Add ACO/RL refinement loop over generated candidates.
- Track whether advantage_over_random improves after each executor or dataset revision.
