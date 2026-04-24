# DTES distance-to-zero analysis

- source: `truth=runs/zeros_100_160_precise.json; dtes=runs/run_20260424_202518/hybrid_colored.json`
- count: `29`
- mean distance: `1.828241683831614`
- median distance: `1.4217828557941061`
- max distance: `7.049904086967942`

## Percentiles

- p50: `1.4217828557941061`
- p75: `2.4455617384600714`
- p90: `4.640826689540861`
- p95: `6.107879202086204`
- p99: `6.932855916409893`
- p100: `7.049904086967942`

## Recommended windows

- recall_0.9000: `window >= 4.640826689540861`
- recall_0.9500: `window >= 6.107879202086204`
- recall_0.9900: `window >= 6.932855916409893`
- recall_1.0000: `window >= 7.049904086967942`

## Window recall sweep

| window | recall | missed |
|---:|---:|---:|
| 0.005 | 0.344828 | 19 |
| 0.01 | 0.344828 | 19 |
| 0.02 | 0.344828 | 19 |
| 0.03 | 0.344828 | 19 |
| 0.04 | 0.344828 | 19 |
| 0.05 | 0.344828 | 19 |
| 0.06 | 0.344828 | 19 |
| 0.08 | 0.344828 | 19 |
| 0.1 | 0.344828 | 19 |
| 0.12 | 0.344828 | 19 |
| 0.15 | 0.344828 | 19 |
| 0.2 | 0.344828 | 19 |

## Interpretation

The minimal hybrid half-window required for a desired recall is the corresponding
quantile of the distance distribution. If the maximum distance is small and stable
across intervals, DTES candidates form a compact cover of the true zero set.