# DTES distance-to-zero analysis

- source: `truth=runs/zeros_100_400_precise.json; dtes=runs/run_20260425_103714/colored_candidates.json`
- count: `173`
- mean distance: `0.9017529778366471`
- median distance: `0.9201680477529521`
- max distance: `3.603265653602705`

## Percentiles

- p50: `0.9201680477529521`
- p75: `1.545606742690154`
- p90: `1.915330299671598`
- p95: `2.6452531026961768`
- p99: `3.1755346069944483`
- p100: `3.603265653602705`

## Recommended windows

- recall_0.9000: `window >= 1.915330299671598`
- recall_0.9500: `window >= 2.6452531026961768`
- recall_0.9900: `window >= 3.1755346069944483`
- recall_1.0000: `window >= 3.603265653602705`

## Window recall sweep

| window | recall | missed |
|---:|---:|---:|
| 0.005 | 0.416185 | 101 |
| 0.01 | 0.416185 | 101 |
| 0.02 | 0.416185 | 101 |
| 0.03 | 0.416185 | 101 |
| 0.04 | 0.416185 | 101 |
| 0.05 | 0.416185 | 101 |
| 0.06 | 0.416185 | 101 |
| 0.08 | 0.416185 | 101 |
| 0.1 | 0.416185 | 101 |
| 0.12 | 0.416185 | 101 |
| 0.15 | 0.416185 | 101 |
| 0.2 | 0.416185 | 101 |

## Interpretation

The minimal hybrid half-window required for a desired recall is the corresponding
quantile of the distance distribution. If the maximum distance is small and stable
across intervals, DTES candidates form a compact cover of the true zero set.