#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Ant-RH full pipeline
# Current repo layout:
#   root/fractal_dtes_aco_zeta_all_zeros_scan.py
#   core/fractal_dtes_crosschannel_explore_eta_clean.py
#   refinement/colored_ants_engine.py
#   refinement/gap_detector.py
#   validation/distance_analysis.py
#   hybrid/hybrid_dtes_guided_scan.py
# ============================================================

# -------------------------
# CONFIG
# -------------------------
T_MIN="${T_MIN:-100}"
T_MAX="${T_MAX:-400}"

N0="${N0:-2500}"
ANTS="${ANTS:-100}"
ITERS="${ITERS:-120}"

TRUTH_STEP="${TRUTH_STEP:-0.01}"
TRUTH_DPS="${TRUTH_DPS:-80}"
DTES_DPS="${DTES_DPS:-50}"

WINDOW="${WINDOW:-0.065}"
HYBRID_STEP="${HYBRID_STEP:-0.01}"
HYBRID_DPS="${HYBRID_DPS:-80}"

GROUPS="${GROUPS:-4}"
ANTS_PER_GROUP="${ANTS_PER_GROUP:-24}"
ITERATIONS_PER_GROUP="${ITERATIONS_PER_GROUP:-20}"
TARGET_COUNT="${TARGET_COUNT:-180}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/run_${RUN_ID}"

TRUTH_PREFIX="runs/zeros_${T_MIN}_${T_MAX}_precise"
TRUTH_FILE="${TRUTH_PREFIX}.json"

mkdir -p runs "$RUN_DIR"

echo "========================================"
echo " Ant-RH / DTES FULL PIPELINE"
echo " RUN:   $RUN_ID"
echo " RANGE: [$T_MIN, $T_MAX]"
echo " OUT:   $RUN_DIR"
echo "========================================"

# -------------------------
# Preflight
# -------------------------
require_file() {
  if [ ! -f "$1" ]; then
    echo "[ERROR] Required file not found: $1"
    exit 1
  fi
}

require_file "fractal_dtes_aco_zeta_all_zeros_scan.py"
require_file "core/fractal_dtes_crosschannel_explore_eta_clean.py"
require_file "refinement/colored_ants_engine.py"
require_file "refinement/gap_detector.py"
require_file "validation/distance_analysis.py"
require_file "hybrid/hybrid_dtes_guided_scan.py"

echo "[OK] Preflight passed"

# -------------------------
# STEP 0: Ground truth
# -------------------------
echo
echo "=== STEP 0: Ground truth zeros ==="

if [ ! -f "$TRUTH_FILE" ]; then
  echo "[INFO] Truth file not found. Generating: $TRUTH_FILE"

  python3 fractal_dtes_aco_zeta_all_zeros_scan.py \
    --t_min "$T_MIN" \
    --t_max "$T_MAX" \
    --step "$TRUTH_STEP" \
    --dps "$TRUTH_DPS" \
    --progress_every 500 \
    --output "$TRUTH_PREFIX"
else
  echo "[OK] Truth file exists: $TRUTH_FILE"
fi

# Copy truth into run dir for reproducibility.
cp "$TRUTH_FILE" "$RUN_DIR/truth.json"

# -------------------------
# STEP 1: DTES core candidate pool
# -------------------------
echo
echo "=== STEP 1: DTES core ==="

python3 core/fractal_dtes_crosschannel_explore_eta_clean.py \
  --t_min "$T_MIN" \
  --t_max "$T_MAX" \
  --n0 "$N0" \
  --ants "$ANTS" \
  --iters "$ITERS" \
  --dps "$DTES_DPS" \
  --output "$RUN_DIR/dtes_candidates.json" \
  --edge_output "$RUN_DIR/dtes_candidates_edgeaware.json" \
  --anchors_output "$RUN_DIR/edge_anchors.json" \
  --metrics "$RUN_DIR/run_metrics_dtes.json"

# -------------------------
# STEP 2: Colored grouped ants refinement
# -------------------------
echo
echo "=== STEP 2: Colored grouped ants ==="

python3 refinement/colored_ants_engine.py \
  --pool "$RUN_DIR/dtes_candidates.json" \
  --output "$RUN_DIR/colored_candidates.json" \
  --metrics "$RUN_DIR/colored_group_metrics.json" \
  --groups "$GROUPS" \
  --ants_per_group "$ANTS_PER_GROUP" \
  --iterations_per_group "$ITERATIONS_PER_GROUP" \
  --target_count "$TARGET_COUNT"

# -------------------------
# STEP 3: Gap detection
# -------------------------
echo
echo "=== STEP 3: Gap detection ==="

python3 refinement/gap_detector.py \
  --candidates "$RUN_DIR/colored_candidates.json" \
  --threshold 1.0 \
  --out "$RUN_DIR/colored_group_gaps.json"

# -------------------------
# STEP 4: Distance analysis for core DTES
# -------------------------
echo
echo "=== STEP 4: Distance analysis: core DTES ==="

python3 validation/distance_analysis.py \
  --truth "$TRUTH_FILE" \
  --dtes "$RUN_DIR/dtes_candidates.json" \
  --t_min "$T_MIN" \
  --t_max "$T_MAX" \
  --out "$RUN_DIR/distance_core"

# -------------------------
# STEP 5: Distance analysis for colored ants
# -------------------------
echo
echo "=== STEP 5: Distance analysis: colored ants ==="

python3 validation/distance_analysis.py \
  --truth "$TRUTH_FILE" \
  --dtes "$RUN_DIR/colored_candidates.json" \
  --t_min "$T_MIN" \
  --t_max "$T_MAX" \
  --out "$RUN_DIR/distance_colored"

# -------------------------
# STEP 6: Hybrid scan from colored candidates
# -------------------------
echo
echo "=== STEP 6: Hybrid scan ==="

python3 hybrid/hybrid_dtes_guided_scan.py \
  --dtes "$RUN_DIR/colored_candidates.json" \
  --t_min "$T_MIN" \
  --t_max "$T_MAX" \
  --window "$WINDOW" \
  --step "$HYBRID_STEP" \
  --dps "$HYBRID_DPS" \
  --progress_every 1 \
  --out "$RUN_DIR/hybrid_colored"

# -------------------------
# STEP 7: Distance analysis for hybrid output
# -------------------------
echo
echo "=== STEP 7: Distance analysis: hybrid zeros ==="

python3 validation/distance_analysis.py \
  --truth "$TRUTH_FILE" \
  --dtes "$RUN_DIR/hybrid_colored.json" \
  --t_min "$T_MIN" \
  --t_max "$T_MAX" \
  --out "$RUN_DIR/distance_hybrid"

# -------------------------
# Summary
# -------------------------
SUMMARY="$RUN_DIR/RUN_SUMMARY.md"

{
  echo "# Ant-RH run summary"
  echo
  echo "- run_id: \`$RUN_ID\`"
  echo "- range: \`[$T_MIN, $T_MAX]\`"
  echo "- truth: \`$TRUTH_FILE\`"
  echo "- window: \`$WINDOW\`"
  echo
  echo "## Outputs"
  echo
  echo "- \`dtes_candidates.json\`"
  echo "- \`dtes_candidates_edgeaware.json\`"
  echo "- \`colored_candidates.json\`"
  echo "- \`colored_group_gaps.json\`"
  echo "- \`distance_core_summary.md\`"
  echo "- \`distance_colored_summary.md\`"
  echo "- \`hybrid_colored.json\`"
  echo "- \`hybrid_colored_stats.json\`"
  echo "- \`distance_hybrid_summary.md\`"
  echo
  echo "## Quick inspect"
  echo
  echo "\`\`\`bash"
  echo "cat $RUN_DIR/distance_core_summary.md"
  echo "cat $RUN_DIR/distance_colored_summary.md"
  echo "cat $RUN_DIR/distance_hybrid_summary.md"
  echo "cat $RUN_DIR/hybrid_colored_stats.json"
  echo "\`\`\`"
} > "$SUMMARY"

echo
echo "========================================"
echo " DONE"
echo " Results: $RUN_DIR"
echo " Summary: $SUMMARY"
echo "========================================"
