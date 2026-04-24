#!/bin/bash
set -e

# =========================
# CONFIG
# =========================
T_MIN=100
T_MAX=400
ANTS=60
ITERS=80
N0=2500
WINDOW=0.065
STEP=0.01
DPS_TRUTH=80
DPS_DTES=50

RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR="runs/run_$RUN_ID"
TRUTH_FILE="runs/zeros_${T_MIN}_${T_MAX}_precise.json"
TRUTH_PREFIX="runs/zeros_${T_MIN}_${T_MAX}_precise"

mkdir -p runs "$RUN_DIR"

echo "========================================"
echo " DTES FULL PIPELINE"
echo " RUN: $RUN_ID"
echo " RANGE: [$T_MIN, $T_MAX]"
echo "========================================"

# =========================
# STEP 0: Generate ground truth if missing
# =========================
echo "=== STEP 0: Ground truth zeros ==="

if [ ! -f "$TRUTH_FILE" ]; then
  echo "[INFO] Truth file not found. Generating: $TRUTH_FILE"

  python3 validation/fractal_dtes_aco_zeta_all_zeros_scan.py \
    --t_min $T_MIN \
    --t_max $T_MAX \
    --step $STEP \
    --dps $DPS_TRUTH \
    --output "$TRUTH_PREFIX"
else
  echo "[OK] Truth file exists: $TRUTH_FILE"
fi

# =========================
# STEP 1: DTES core
# =========================
echo "=== STEP 1: DTES core ==="

python3 core/fractal_dtes_crosschannel_explore_eta_clean.py \
  --t_min $T_MIN \
  --t_max $T_MAX \
  --n0 $N0 \
  --ants $ANTS \
  --iters $ITERS \
  --dps $DPS_DTES \
  --output "$RUN_DIR/dtes_candidates.json" \
  --edge_output "$RUN_DIR/dtes_candidates_edgeaware.json" \
  --metrics "$RUN_DIR/run_metrics.json"

# =========================
# STEP 2: Colored ants refinement
# =========================
echo "=== STEP 2: Colored ants refinement ==="

python3 refinement/colored_ants_engine.py \
  --pool "$RUN_DIR/dtes_candidates.json" \
  --output "$RUN_DIR/colored_candidates.json" \
  --metrics "$RUN_DIR/colored_metrics.json" \
  --groups 4 \
  --ants_per_group 24 \
  --iterations_per_group 20 \
  --target_count 180

# =========================
# STEP 3: Gap detection
# =========================
echo "=== STEP 3: Gap detection ==="

python3 refinement/gap_detector.py \
  --candidates "$RUN_DIR/colored_candidates.json" \
  --threshold 1.0 \
  --out "$RUN_DIR/gaps.json"

# =========================
# STEP 4: Compare colored DTES vs truth
# =========================
echo "=== STEP 4: Compare vs truth ==="

python3 validation/compare_dtes_vs_truth.py \
  --truth "$TRUTH_FILE" \
  --dtes "$RUN_DIR/colored_candidates.json" \
  --tol $WINDOW \
  --t_min $T_MIN \
  --t_max $T_MAX \
  --out "$RUN_DIR/compare_colored"

# =========================
# STEP 5: Distance analysis
# =========================
echo "=== STEP 5: Distance analysis ==="

python3 validation/distance_analysis.py \
  --truth "$TRUTH_FILE" \
  --dtes "$RUN_DIR/colored_candidates.json" \
  --t_min $T_MIN \
  --t_max $T_MAX \
  --out "$RUN_DIR/distance_colored"

# =========================
# STEP 6: Hybrid scan
# =========================
echo "=== STEP 6: Hybrid scan ==="

python3 hybrid/hybrid_dtes_guided_scan.py \
  --dtes "$RUN_DIR/colored_candidates.json" \
  --t_min $T_MIN \
  --t_max $T_MAX \
  --window $WINDOW \
  --step $STEP \
  --dps $DPS_TRUTH \
  --out "$RUN_DIR/hybrid_colored"

# =========================
# STEP 7: Compare hybrid zeros vs truth
# =========================
echo "=== STEP 7: Compare hybrid vs truth ==="

python3 validation/compare_dtes_vs_truth.py \
  --truth "$TRUTH_FILE" \
  --dtes "$RUN_DIR/hybrid_colored.json" \
  --tol $STEP \
  --t_min $T_MIN \
  --t_max $T_MAX \
  --out "$RUN_DIR/compare_hybrid"

# =========================
# DONE
# =========================
echo "========================================"
echo " DONE"
echo " Results saved in: $RUN_DIR"
echo " Truth file: $TRUTH_FILE"
echo "========================================"
