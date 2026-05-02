# Ant-RH NCG + Braid integration layer

This bundle adds a small experimental **noncommutative-geometry / braid spectral layer** to Ant-RH.

It gives you:

- finite truncation of `l2(B_n)` from braid words;
- braid length / exponential-growth Dirac-style operator `D_B`;
- symmetric adjacency coupling between braid states;
- safe Hermitian eigenvalue computation;
- spectral loss against zeta zero ordinates;
- heat-trace / spectral-zeta diagnostics;
- smoke CLI and pytest tests.

## Copy into Ant-RH

From Ant-RH repo root:

```bash
cp -r ant_rh_ncg_integration/core/ncg_braid_spectral.py core/
cp -r ant_rh_ncg_integration/validation/run_ncg_braid_smoke.py validation/
cp -r ant_rh_ncg_integration/configs/ncg_braid_smoke.yaml configs/
cp -r ant_rh_ncg_integration/tests/test_ncg_braid_spectral.py tests/
```

Or unzip this archive in the repo root and move files accordingly.

## Run smoke test

```bash
python3 validation/run_ncg_braid_smoke.py \
  --n_strands 4 \
  --max_word_len 8 \
  --dim 128 \
  --device cpu \
  --out_dir runs/ncg_braid_smoke
```

For MPS on Mac:

```bash
python3 validation/run_ncg_braid_smoke.py --device mps --dtype float32
```

If MPS eigensolver fails, use CPU/float64 for spectral validation.

## Optional zeros file

```bash
python3 validation/run_ncg_braid_smoke.py --zeros data/zeta_zeros.txt
```

Accepted: txt/csv/json with a list of ordinates.

## How to plug into existing Ant-RH training

Inside an ACO/RL scoring step:

```python
from core.ncg_braid_spectral import (
    BraidNCGConfig,
    enumerate_braid_words,
    build_dirac_operator,
    load_zeros,
    full_ncg_loss,
)

cfg = BraidNCGConfig(n_strands=4, max_word_len=8, dim=128, device="cpu")
words = enumerate_braid_words(cfg.n_strands, cfg.max_word_len, cfg.dim)
zeros = load_zeros("data/zeta_zeros.txt", cfg)
D = build_dirac_operator(words, cfg)
loss, stats = full_ncg_loss(D, zeros, cfg)
```

Then add `loss` to your candidate reward:

```python
reward = -float(loss.detach().cpu())
```

## Theoretical interpretation

This approximates a finite spectral system:

```text
A_B  = C[B_n] or finite braid operator grammar
H_B  = l2(admissible braid words)
D_B  = diagonal braid-length energy + symmetric transition operator
zeta_D(s) = Tr(|D_B|^{-s})
```

The finite truncation is only a numerical proxy. It does **not** prove RH. It is useful as a stricter operator-search prior for Ant-RH.
