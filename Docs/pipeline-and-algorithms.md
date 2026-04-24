# Pipeline and algorithms

This page ties the **code** to the **math** in the root [README.md](../README.md). It is not a proof of the Riemann Hypothesis; it documents a **computational experiment** pipeline.

## Mathematical objects

- **Zeta:** \(\zeta(\tfrac12 + it)\) evaluated with `mpmath`.
- **Hardy \(Z(t)\):** real-valued on the critical line; nontrivial zeros of \(\zeta\) on \(\mathrm{Re}(s)=\tfrac12\) correspond to zeros of \(Z\) (under standard indexing assumptions in the implemented range).

## Pipeline stages (code mapping)

```text
Ground truth Hardy-Z scan     →  fractal_dtes_aco_zeta_all_zeros_scan.py
        ↓
DTES core candidate discovery →  core/fractal_dtes_crosschannel_explore_eta_clean.py
        ↓                         (or full fractal+ACO: run_crosschannel_live_v2.py)
Optional colored-ant refinement → refinement/colored_ants_engine.py
        ↓
Hybrid local scan             →  hybrid/hybrid_dtes_guided_scan.py
        ↓
Distance / recall analysis    →  validation/distance_analysis.py
```

The bash script [run_full_pipeline.sh](../run_full_pipeline.sh) automates this with sensible defaults and writes under `runs/run_<id>/`.

## DTES (conceptual)

The README defines an energy \(E(t) = -\log|\zeta(\tfrac12+it)|\) and a tropical surrogate. In code, energies and scores are **discrete**: coarse grid values, node energies on a dyadic tree, multiscale features, and ACO path scores, depending on which module you run.

Two implementation tracks:

1. **`fractal_dtes_aco_zeta_metrics.py` / `_visual` / `_eta` / `_crosschannel`:** full tree + graph + multi-agent ACO.
2. **`core/fractal_dtes_crosschannel_explore_eta_clean.py`:** grid-based multi-channel pool, exploration-aware binning, local refine; **default in `run_full_pipeline.sh`**.

## Hybrid recovery (intuition)

If every true zero lies within \(\delta\) of some DTES candidate, scanning width \(w \ge \delta\) around each candidate (after merging overlapping windows) recovers all zeros inside the interval **up to numerical tolerance**. The hybrid script realizes window merge + dense scan + bisection.

## Complexity (engineering view)

- **Full scan:** proportional to number of \(t\) samples on the whole interval.
- **Hybrid:** proportional to total length of merged windows \(\times\) step density; wins when candidates are sparse and well placed.

See README tables for a reported example on \([100,160]\).

## Numerical caveats

- Precision `--dps` trades CPU for reliability near oscillatory regions.
- Very coarse grids or tiny hybrid `--window` can miss zeros (violates the \(\delta\) assumption).
- Edge anchors extend coverage near interval boundaries; they are not substitutes for core DTES when measuring candidate quality.
