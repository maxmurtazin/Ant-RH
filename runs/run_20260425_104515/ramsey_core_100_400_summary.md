# Ramsey graph analysis

- nodes: `173`
- edges: `707`
- max clique size: `9`
- max monochromatic clique size: `9`
- Ramsey score: `0.052023`

## Color stats

| color | edges | max clique | triangles | clustering |
|---|---:|---:|---:|---:|
| red | 157 | 4 | 29 | 0.2732 |
| blue | 324 | 4 | 138 | 0.3463 |
| green | 120 | 9 | 233 | 0.8324 |
| violet | 106 | 2 | 0 | 0.0000 |

## Shuffled coloring baseline

- permutations: `500`
- mean max mono clique: `4.114`
- std max mono clique: `0.317811264746862`
- p95 max mono clique: `5.0`

## Interpretation

- Red clique: energetically coherent region.
- Blue clique: local-road coherence.
- Green clique: boundary concentration.
- Violet clique: bridge/gap structure.
- If real max monochromatic clique exceeds shuffled baseline, coloring is non-random.