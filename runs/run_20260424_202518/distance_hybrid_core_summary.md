# DTES distance-to-zero analysis

- source: `truth=runs/zeros_100_160_precise.json; dtes=runs/run_20260424_202518/hybrid_core.json`
- count: `29`
- mean distance: `0.0`
- median distance: `0.0`
- max distance: `0.0`

## Percentiles

- p50: `0.0`
- p75: `0.0`
- p90: `0.0`
- p95: `0.0`
- p99: `0.0`
- p100: `0.0`

## Recommended windows

- recall_0.9000: `window >= 0.0`
- recall_0.9500: `window >= 0.0`
- recall_0.9900: `window >= 0.0`
- recall_1.0000: `window >= 0.0`

## Window recall sweep

| window | recall | missed |
|---:|---:|---:|
| 0.005 | 1.000000 | 0 |
| 0.01 | 1.000000 | 0 |
| 0.02 | 1.000000 | 0 |
| 0.03 | 1.000000 | 0 |
| 0.04 | 1.000000 | 0 |
| 0.05 | 1.000000 | 0 |
| 0.06 | 1.000000 | 0 |
| 0.08 | 1.000000 | 0 |
| 0.1 | 1.000000 | 0 |
| 0.12 | 1.000000 | 0 |
| 0.15 | 1.000000 | 0 |
| 0.2 | 1.000000 | 0 |

## Interpretation

The minimal hybrid half-window required for a desired recall is the corresponding
quantile of the distance distribution. If the maximum distance is small and stable
across intervals, DTES candidates form a compact cover of the true zero set.