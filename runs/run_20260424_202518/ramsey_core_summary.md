# Ramsey graph analysis

- nodes: `29`
- edges: `93`
- max clique size: `7`
- max monochromatic clique size: `4`
- Ramsey score: `0.137931`

## Color stats

| color | edges | max clique | triangles | clustering |
|---|---:|---:|---:|---:|
| red | 26 | 4 | 10 | 0.5422 |
| blue | 36 | 4 | 15 | 0.5050 |
| green | 14 | 4 | 7 | 0.8222 |
| violet | 17 | 2 | 0 | 0.0000 |

## Shuffled coloring baseline

- permutations: `100`
- mean max mono clique: `3.2`
- std max mono clique: `0.4`
- p95 max mono clique: `4.0`

## Interpretation

- Red clique: energetically coherent region.
- Blue clique: local-road coherence.
- Green clique: boundary concentration.
- Violet clique: bridge/gap structure.
- If real max monochromatic clique exceeds shuffled baseline, coloring is non-random.