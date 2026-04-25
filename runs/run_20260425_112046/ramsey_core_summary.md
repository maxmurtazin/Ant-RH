# Ramsey graph analysis

- nodes: `150`
- edges: `614`
- max clique size: `9`
- max monochromatic clique size: `9`
- Ramsey score: `0.060000`

## Color stats

| color | edges | max clique | triangles | clustering |
|---|---:|---:|---:|---:|
| red | 144 | 4 | 30 | 0.3392 |
| blue | 274 | 4 | 113 | 0.3464 |
| green | 107 | 9 | 217 | 0.8585 |
| violet | 89 | 2 | 0 | 0.0000 |

## Shuffled coloring baseline

- permutations: `500`
- mean max mono clique: `4.062`
- std max mono clique: `0.25720808696462094`
- p95 max mono clique: `5.0`

## Interpretation

- Red clique: energetically coherent region.
- Blue clique: local-road coherence.
- Green clique: boundary concentration.
- Violet clique: bridge/gap structure.
- If real max monochromatic clique exceeds shuffled baseline, coloring is non-random.