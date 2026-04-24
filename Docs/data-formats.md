# Data formats

Convention: everywhere on the critical line, \(s = \tfrac12 + it\); stored values are the imaginary part **\(t\)** (real float).

## Truth zeros (ground scan output)

Produced by `fractal_dtes_aco_zeta_all_zeros_scan.py` as `<prefix>.json`.

Top-level object:

```json
{
  "config": { "...": "cli args" },
  "count": 29,
  "zeros": [ { "t": 110.123, "abs_zeta": 1e-15, "...": "..." } ],
  "runtime_s": 123.4
}
```

Each zero entry includes at least: `index_in_run`, `t`, `abs_zeta`, `hardy_z`, bracket metadata, `method`.

For **distance analysis**, pass this file as `--truth`. The loader accepts several JSON layouts; lists of objects with a `t` field are recognized.

## DTES / candidate JSON

### Canonical “candidates list” wrapper

Many tools expect a **dict** with a `candidates` array:

```json
{
  "candidates": [
    { "rank": 1, "t": 110.5, "score": 1.2, "source": "..." }
  ],
  "count": 1
}
```

Items may be bare floats in some generators; hybrid and distance loaders normalize via `extract_t()`.

### Core vs edge-aware (important for metrics)

- **Core DTES:** only algorithmically chosen centers (e.g. `run_crosschannel_live_v2.py --output`, or `core/..._clean.py --output`).
- **Edge-aware:** core plus **boundary anchors** near `t_min` / `t_max` for hybrid coverage (`--edge_output`).

For **DTES quality** (distance to true zeros), analyze **core** files, not edge-aware-only blends, unless you explicitly want boundary behavior in the metric.

## Hybrid scan output

`hybrid/hybrid_dtes_guided_scan.py` writes `<out>.json`, `.csv`, `.txt`, `<out>_stats.json`, `<out>_windows.csv`.

The JSON lists recovered zeros in a format compatible with `distance_analysis.py` `--dtes` (sequence of \(t\) or objects with `t`).

## Metrics / run logs

- `run_metrics*.json`, `metrics_summary.json`: timings, config, optional `aco_history` embedded or sidecar.
- `aco_history.json`: per-iteration records (scores, ETA, diversity) when produced by ETA or live runners.

## Colored ants

`refinement/colored_ants_engine.py` writes a candidate JSON plus `colored_group_metrics.json` (group statistics for the run).

## Gap detector

`refinement/gap_detector.py` outputs JSON describing gaps larger than `--threshold` in sorted candidate \(t\).
