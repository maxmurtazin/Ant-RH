# Ramsey graph analysis

- nodes: `10`
- edges: `37`
- max clique size: `7`
- max monochromatic clique size: `4`
- Ramsey score: `0.400000`

## Color stats

| color | edges | max clique | triangles | clustering |
|---|---:|---:|---:|---:|
| red | 12 | 4 | 8 | 0.8389 |
| blue | 16 | 3 | 7 | 0.5132 |
| green | 1 | 2 | 0 | 0.0000 |
| violet | 8 | 2 | 0 | 0.0000 |

## Shuffled coloring baseline

- permutations: `100`
- mean max mono clique: `3.22`
- std max mono clique: `0.4142463035441596`
- p95 max mono clique: `4.0`

## Interpretation

- Red clique: energetically coherent region.
- Blue clique: local-road coherence.
- Green clique: boundary concentration.
- Violet clique: bridge/gap structure.
- If real max monochromatic clique exceeds shuffled baseline, coloring is non-random.