# Ant-RH: DTES-Guided Discovery of Riemann Zeta Zeros

## Abstract

We propose a computational framework for locating non-trivial zeros of the Riemann zeta function on the critical line using:

- Deformable Tropical Energy Space (DTES)
- Ant Colony Optimization (ACO)
- Hybrid local refinement via windowed root-finding

The method replaces uniform scanning with structured exploration of an implicit energy landscape.

---

## Problem formulation

We seek zeros of the Riemann zeta function on the critical line:

$$
\zeta\left(\frac{1}{2} + it\right) = 0.
$$

Equivalently, using the Hardy Z-function:

$$
Z(t) = e^{i\theta(t)}\zeta\left(\frac{1}{2} + it\right) \in \mathbb{R}.
$$

Zeros correspond to sign changes:

$$
Z(t_k) = 0,
\qquad
Z(t_{k}^{-})Z(t_{k}^{+}) < 0.
$$

---

## DTES formulation

We define an energy landscape:

$$
E(t) = -\log\left|\zeta\left(\frac{1}{2} + it\right)\right|.
$$

DTES constructs a piecewise-linear tropical surrogate:

$$
E_{\mathrm{trop}}(t)
=
\max_i (a_i t + b_i).
$$

Zeros are interpreted as high-energy attractors or structural transition points in this landscape.

---

## Hybrid recovery theorem

Let:

- $\mathcal{Z}$ be the true zero set;
- $\mathcal{C}$ be the DTES candidate set;
- $w > 0$ be the hybrid scan half-window.

If:

$$
\forall z \in \mathcal{Z},
\quad
\exists c \in \mathcal{C}
\quad
\text{such that}
\quad
|z-c| \leq \delta,
$$

then for any $w \geq \delta$, the hybrid scan over

$$
\bigcup_{c \in \mathcal{C}} [c-w,c+w]
$$

recovers all zeros in $\mathcal{Z}$.

---

## Experimental result

Fast validation interval:

$$
[100,160]
$$

| Metric | Value |
|---|---:|
| True zeros | 29 |
| DTES candidates | 29 |
| Hybrid recovered | 29 |
| Recall | 1.0 |
| Scanned fraction | 0.077 |
| Estimated speedup | 10–13x |

---

## Complexity

Full scan:

$$
T_{\mathrm{full}} = O(N).
$$

Hybrid DTES-guided scan:

$$
T_{\mathrm{hybrid}}
=
O\left(\sum_i \frac{2w_i}{h}\right),
$$

where:

- $w_i$ is the local scan half-window;
- $h$ is the scan step.

If the DTES candidate set is sparse and accurate, then:

$$
T_{\mathrm{hybrid}} \ll T_{\mathrm{full}}.
$$

---

## Pipeline

```text
Ground truth Hardy-Z scan
        ↓
DTES core candidate discovery
        ↓
Optional colored-ant refinement
        ↓
Hybrid local scan
        ↓
Distance / recall analysis
```

---

## V6 Direct Operator Learning

V6 adds an experimental direct operator learning path. Given exported DTES
spectral input, it initializes a self-adjoint operator from the DTES graph and
then learns a symmetric operator \(H_\theta\) whose eigenvalues are compared
against supplied zeta-zero ordinates.

```bash
python3 validation/train_dtes_operator.py \
  --input runs/dtes_spectral_input.json \
  --out runs/operator_learning_report.json \
  --steps 2000 \
  --lr 1e-3 \
  --k 50
```

The run writes:

- `runs/operator_learning_report.json`
- `runs/learned_eigenvalues.csv`
- `runs/learned_operator.npy`
- `runs/loss_curve.csv`

If `matplotlib` is available, it also writes optional diagnostic plots for the
loss curve, learned spectrum, and spacing distribution.

This is a numerical Hilbert-Polya-inspired diagnostic and fitting experiment.
It does not constitute, and should not be interpreted as, a proof of the
Riemann hypothesis.

---

## Documentation

Extended guides (setup, repo layout, CLI flags, JSON formats): [Docs/README.md](Docs/README.md).
