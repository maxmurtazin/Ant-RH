# Repository layout

## Top-level

| Path | Role |
|------|------|
| [README.md](../README.md) | Mathematical framing, abstract, pipeline diagram |
| [requirements.txt](../requirements.txt) | Python dependencies |
| [run_full_pipeline.sh](../run_full_pipeline.sh) | End-to-end bash pipeline into `runs/` |

### Root Python entry points

| Script | Purpose |
|--------|---------|
| `fractal_dtes_aco_zeta_all_zeros_scan.py` | Ground-truth zeros via Hardy \(Z\) sign scan + bisection |
| `fractal_dtes_aco_zeta_metrics.py` | Core `FractalDTESACOZeta`, `ZetaSearchConfig`, grid/tree/ACO |
| `fractal_dtes_aco_zeta_visual.py` | Matplotlib layer on top of metrics |
| `fractal_dtes_aco_zeta_eta.py` | ETA logging, ACO history, optional early stop |
| `fractal_dtes_aco_zeta_crosschannel.py` | Self-contained CrossChannel variant (large, duplicated config) |
| `run_crosschannel_live_v2.py` | **Preferred** CrossChannel CLI: separates core vs edge-aware JSON |
| `run_crosschannel_live.py` | Older live runner (see v2 for cleaner outputs) |
| `run_crosschannel_fixed.py` | Thin CLI over `fractal_dtes_aco_zeta_crosschannel` |
| `run_with_result_figures.py` | ETA run + `figures_from_results` |
| `fractal_dtes_aco_zeta_crosschannel_autosave.py` | Minimal demo (grid minima of \(|\zeta|\)), not full DTES |

## `core/`

| File | Purpose |
|------|---------|
| `fractal_dtes_crosschannel_explore_eta_clean.py` | **Pipeline default** for DTES candidates: grid, multi-channel scoring, exploration-aware selection, refine, core + edge-aware JSON. Does not import `fractal_dtes_aco_zeta_crosschannel.py`. |

## `hybrid/`

| File | Purpose |
|------|---------|
| `hybrid_dtes_guided_scan.py` | Merge windows around DTES candidates, scan Hardy \(Z\) only inside windows, output zeros + stats |

## `validation/`

| File | Purpose |
|------|---------|
| `distance_analysis.py` | Compare truth vs candidates: summaries, CSV, optional plots |
| `fractal_dtes_aco_zeta_all_zeros_scan.py` | Same logic as root truth scanner (for scripts that `cd` here) |
| `figures_from_results.py` | Figures from `metrics_summary.json` / `aco_history.json` |
| `validate_zeros_and_spacing_eta.py` | ETA-heavy validator (needs external `validate_zeros_and_spacing`) |

## `refinement/`

| File | Purpose |
|------|---------|
| `colored_ants_engine.py` | Grouped colored ants on a candidate pool JSON |
| `gap_detector.py` | Large gaps between sorted candidate \(t\) values |
| `dynamic_roads.py` | Formula reference for dynamic road / pheromone terms |

## `runs/`

Experiment outputs (truth files, per-run directories). Safe to delete or archive; not required for the library code to exist.

## `archive_old/`

Historical patch snippets and backups. **Not** part of the supported pipeline; do not rely on it for production runs.
