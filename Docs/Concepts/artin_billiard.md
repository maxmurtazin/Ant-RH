# Artin Billiard

The Artin billiard layer is the symbolic and geometric representation built from Artin words over PSL(2,Z)-style generators.

- `core/artin_symbolic_billiard.py` is the main implementation entry point.
- Words encode discrete actions that later feed operator construction and search.
- Hyperbolic properties, traces, and lengths are extracted from these words.
- ACO and RL operate over this symbolic space.
- The current system assumes these words should matter to the operator, but the sensitivity diagnostic shows that this dependence is weak in the present implementation.
- That weak coupling is one of the main reasons current search quality is limited.
# Artin Billiard

Artin billiard is the symbolic/geometric layer based on PSL(2,Z) words and associated hyperbolic data.

## Role in Ant-RH
- Generates symbolic words used by operator and search modules.
- Connects geometric structure to Selberg-style constraints.
