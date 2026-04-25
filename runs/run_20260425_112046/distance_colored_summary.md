# DTES distance-to-zero analysis

- source: `truth=runs/zeros_150_450_precise.json; dtes=runs/run_20260425_112046/colored_candidates.json`
- count: `183`
- mean distance: `0.8582837929495275`
- median distance: `0.897945659701918`
- max distance: `3.031247907511414`

## Percentiles

- p50: `0.897945659701918`
- p75: `1.495911322652475`
- p90: `2.0049673010332985`
- p95: `2.3366562618939044`
- p99: `2.835926185655973`
- p100: `3.031247907511414`

## Recommended windows

- recall_0.9000: `window >= 2.0049673010332985`
- recall_0.9500: `window >= 2.3366562618939044`
- recall_0.9900: `window >= 2.835926185655973`
- recall_1.0000: `window >= 3.031247907511414`

## Window recall sweep

| window | recall | missed |
|---:|---:|---:|
| 0.005 | 0.409836 | 108 |
| 0.01 | 0.409836 | 108 |
| 0.02 | 0.409836 | 108 |
| 0.03 | 0.409836 | 108 |
| 0.04 | 0.409836 | 108 |
| 0.05 | 0.409836 | 108 |
| 0.06 | 0.409836 | 108 |
| 0.08 | 0.409836 | 108 |
| 0.1 | 0.409836 | 108 |
| 0.12 | 0.409836 | 108 |
| 0.15 | 0.409836 | 108 |
| 0.2 | 0.409836 | 108 |

## Interpretation

The minimal hybrid half-window required for a desired recall is the corresponding
quantile of the distance distribution. If the maximum distance is small and stable
across intervals, DTES candidates form a compact cover of the true zero set.