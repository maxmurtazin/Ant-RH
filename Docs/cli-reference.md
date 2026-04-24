# CLI reference

All commands assume the **repository root** as the current working directory unless noted.

---

## Ground truth: Hardy \(Z\) scan

**Script:** `fractal_dtes_aco_zeta_all_zeros_scan.py`  
**Same interface:** `validation/fractal_dtes_aco_zeta_all_zeros_scan.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--t_min` | float | required | Interval start (Gram point \(t\)) |
| `--t_max` | float | required | Interval end |
| `--step` | float | `0.01` | Uniform scan step |
| `--dps` | int | `80` | `mpmath` decimal precision |
| `--tol_t` | float | `1e-12` | Merge tolerance in \(t\) |
| `--zero_value_tol` | float | `1e-30` | Treat \(|Z|\) as zero |
| `--max_bisect_iter` | int | `120` | Bisection cap per bracket |
| `--progress_every` | int | `500` | Progress print cadence |
| `--output` | str | required | Output **prefix** (no extension); writes `.json`, `.csv`, `.txt` |

---

## DTES core (pipeline default)

**Script:** `core/fractal_dtes_crosschannel_explore_eta_clean.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--t_min`, `--t_max` | float | required | Search interval |
| `--n0` | int | `2500` | Grid points |
| `--ants` | int | `100` | Scales default target candidate count |
| `--iters` | int | `120` | CLI compatibility; selection is deterministic |
| `--dps` | int | `50` | mpmath precision |
| `--target_count` | int | auto | Cap selected pool size |
| `--coverage_bins` | int | `64` | Exploration bins along \(t\) |
| `--exploration_strength` | float | `1.0` | Under-filled bin bonus |
| `--refine_radius` | float | auto | Local refine window |
| `--merge_tol` | float | `1e-6` | Merge nearby refined \(t\) |
| `--edge_padding`, `--edge_step` | float | `2.5`, `0.05` | Edge anchor strip |
| `--no_tqdm` | flag | off | Disable tqdm |
| `--output` | str | `dtes_candidates_explore.json` | **Core** candidates only |
| `--edge_output` | str | `…_edgeaware.json` | Core + edge anchors |
| `--anchors_output` | str | `edge_anchors.json` | Anchors only |
| `--metrics` | str | `run_metrics_explore.json` | Run metrics JSON |

---

## CrossChannel live (full fractal + ACO)

**Preferred:** `run_crosschannel_live_v2.py` (core vs edge-aware files separated)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--t_min`, `--t_max` | float | required | Interval |
| `--n0` | int | `2500` | Grid size |
| `--levels` | int | `8` | Dyadic tree depth |
| `--feature_levels` | int | `5` | Multiscale feature levels |
| `--ants`, `--iters` | int | `60`, `80` | ACO |
| `--max_ant_steps` | int | `20` | Steps per ant path |
| `--top_candidate_nodes` | int | `256` | Ranked leaves cap |
| `--r0` | float | `6.0` | Base feature radius |
| `--dps` | int | `50` | Precision |
| `--edge_padding`, `--edge_step` | float | `2.5`, `0.05` | Edge anchors |
| `--progress_every` | int | `1` | Text progress |
| `--no_tqdm` | flag | off | |
| `--output` | str | `dtes_candidates.json` | **Core DTES only** |
| `--edge_output` | str | `dtes_candidates_edgeaware.json` | Core + anchors |
| `--anchors_output` | str | `edge_anchors.json` | Anchors only |
| `--metrics` | str | `run_metrics.json` | Timings + ACO history |

**Also:** `run_crosschannel_live.py` (similar; older semantics), `run_crosschannel_fixed.py` (single combined output file).

---

## Hybrid guided scan

**Script:** `hybrid/hybrid_dtes_guided_scan.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dtes` | path | required | Candidates JSON |
| `--t_min`, `--t_max` | float | required | Clip interval |
| `--window` | float | `0.2` | Half-window per candidate |
| `--top_k` | int | none | Use only top-\(k\) candidates |
| `--step` | float | `0.01` | Scan step inside windows |
| `--dps` | int | `80` | Precision |
| `--zero_value_tol` | float | `1e-30` | |
| `--verify_abs_zeta` | float | `1e-8` | Post-refine check |
| `--max_bisect_iter` | int | `120` | |
| `--merge_tol` | float | `1e-7` | |
| `--progress_every` | int | `1` | |
| `--out` | str | `hybrid_zeros` | Output prefix |

---

## Distance analysis

**Script:** `validation/distance_analysis.py`

| Argument | Type | Description |
|----------|------|-------------|
| `--truth` | path | Truth zeros JSON |
| `--dtes` | path | Candidates JSON |
| `--t_min`, `--t_max` | float | Optional filter |
| `--distances_csv` | path | Alternative: precomputed CSV |
| `--target_recalls` | str | Comma-separated recalls for window sweep |
| `--window_sweep` | str | Comma-separated half-windows |
| `--out` | str | Output prefix for `_summary.md`, `_stats.json`, plots |
| `--no_plots` | flag | Skip PNG |

---

## Refinement

**`refinement/colored_ants_engine.py`:** `--pool`, `--output`, `--metrics`, `--groups`, `--ants_per_group`, `--iterations_per_group`, `--max_steps`, `--target_count`.

**`refinement/gap_detector.py`:** `--candidates`, `--threshold`, `--out`.

---

## Figures from ETA run

**Script:** `validation/figures_from_results.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--results` | `fractal_dtes_aco_eta_output` | Directory with `metrics_summary.json` / `aco_history.json` |
| `--out` | `paper_figures_from_results` | Output directory |

---

## `run_with_result_figures.py`

| Argument | Default |
|----------|---------|
| `--out` | `fractal_dtes_aco_eta_output` |
| `--figures` | `paper_figures_from_results` |
| `--t-min`, `--t-max` | `10`, `40` |
| `--n-grid` | `2048` |
| `--tree-depth` | `8` |
| `--feature-levels` | `5` |
| `--n-ants` | `48` |
| `--n-iterations` | `60` |
| `--early-stop-patience` | `0` |

---

## Minimal demo autosave

**`fractal_dtes_aco_zeta_crosschannel_autosave.py`:** `--t_min`, `--t_max`, `--n0`, `--output`.
