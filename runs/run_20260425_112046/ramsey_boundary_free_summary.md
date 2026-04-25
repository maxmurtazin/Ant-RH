# Boundary-free Ramsey graph analysis

- nodes: `183`
- edges: `741`
- cyclic topology: `True`
- max clique size: `6`
- max monochromatic clique size: `4`
- Ramsey score: `0.021858`

## Color stats

| color | edges | max clique | triangles | clustering |
|---|---:|---:|---:|---:|
| red | 207 | 4 | 48 | 0.3348 |
| blue | 385 | 4 | 159 | 0.3311 |
| violet | 149 | 2 | 0 | 0.0000 |

## Shuffled coloring baseline

- permutations: `500`
- mean max mono clique: `4.276`
- std max mono clique: `0.44701677820860364`
- p95 max mono clique: `5.0`
- color p95: `{'red': 4.0, 'blue': 5.0, 'violet': 3.0}`

## Interpretation

This analysis removes explicit boundary coloring. If the previous green K9 disappears,
the earlier Ramsey signal was mostly finite-window boundary structure. If a large
red/blue/violet clique remains above shuffled baseline, it is stronger evidence for
intrinsic DTES/Ramsey organization in the bulk.