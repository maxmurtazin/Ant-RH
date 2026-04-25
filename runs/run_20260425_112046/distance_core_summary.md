# DTES distance-to-zero analysis

- source: `truth=runs/zeros_150_450_precise.json; dtes=runs/run_20260425_112046/dtes_candidates.json`
- count: `183`
- mean distance: `2.360710291487109e-14`
- median distance: `0.0`
- max distance: `5.684341886080802e-14`

## Percentiles

- p50: `0.0`
- p75: `5.684341886080802e-14`
- p90: `5.684341886080802e-14`
- p95: `5.684341886080802e-14`
- p99: `5.684341886080802e-14`
- p100: `5.684341886080802e-14`

## Recommended windows

- recall_0.9000: `window >= 5.684341886080802e-14`
- recall_0.9500: `window >= 5.684341886080802e-14`
- recall_0.9900: `window >= 5.684341886080802e-14`
- recall_1.0000: `window >= 5.684341886080802e-14`

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