# DTES Colored Group Ants v3 Bundle

This bundle extends the DTES zeta-zero pipeline with:

1. **Clean exploration runner**
   - `fractal_dtes_crosschannel_explore_eta_clean.py`
   - produces a broad DTES candidate pool with ETA and edge-aware outputs.

2. **Colored grouped ants**
   - `colored_ants_engine.py`
   - colors:
     - red = exploitation
     - blue = exploration
     - green = boundary-aware
     - violet = gap-bridge

3. **Dynamic roads**
   - `dynamic_roads.py`
   - formula reference for dynamic branch roads.

4. **Gap detector**
   - `gap_detector.py`
   - finds uncovered intervals between candidates.

## Step 1. Generate DTES pool

```bash
python3 fractal_dtes_crosschannel_explore_eta_clean.py \
  --t_min 100 --t_max 400 \
  --n0 2500 \
  --ants 100 \
  --iters 120 \
  --dps 50 \
  --output dtes_candidates_explore.json \
  --edge_output dtes_candidates_explore_edgeaware.json \
  --metrics run_metrics_explore.json
```

## Step 2. Run colored grouped ants

```bash
python3 colored_ants_engine.py \
  --pool dtes_candidates_explore.json \
  --output colored_group_candidates.json \
  --metrics colored_group_metrics.json \
  --groups 4 \
  --ants_per_group 24 \
  --iterations_per_group 20 \
  --target_count 180
```

## Step 3. Detect gaps

```bash
python3 gap_detector.py \
  --candidates colored_group_candidates.json \
  --threshold 1.0 \
  --out colored_group_gaps.json
```

## Step 4. Validate distance to true zeros

```bash
python3 distance_analysis.py \
  --truth zeros_100_400_precise.json \
  --dtes colored_group_candidates.json \
  --t_min 100 --t_max 400 \
  --out distance_colored_group
```

## Step 5. Hybrid scan

```bash
python3 hybrid_dtes_guided_scan.py \
  --dtes colored_group_candidates.json \
  --t_min 100 --t_max 400 \
  --window 0.065 \
  --step 0.01 \
  --dps 80 \
  --out hybrid_colored_group_100_400
```

## Hypothesis v2

Grouped colored ants reduce mode collapse by separating policies:
- red ants exploit low-energy attractors;
- blue ants force exploration;
- green ants protect boundaries;
- violet ants bridge uncovered gaps.

Dynamic roads make branch weights adaptive rather than fixed.
