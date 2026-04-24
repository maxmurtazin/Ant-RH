#!/bin/bash

set -e

echo "=== Creating structure ==="

mkdir -p core refinement validation hybrid tools archive_old runs figures

echo "=== Moving core ==="
mv -n fractal_dtes_crosschannel_explore_eta_clean.py core/ 2>/dev/null || true

echo "=== Moving refinement ==="
mv -n colored_ants_engine.py refinement/ 2>/dev/null || true
mv -n dynamic_roads.py refinement/ 2>/dev/null || true
mv -n gap_detector.py refinement/ 2>/dev/null || true

echo "=== Moving validation ==="
mv -n distance_analysis.py validation/ 2>/dev/null || true
mv -n validate_zeros_and_spacing_eta.py validation/ 2>/dev/null || true
mv -n figures_from_results.py validation/ 2>/dev/null || true

echo "=== Moving hybrid ==="
mv -n hybrid_dtes_guided_scan.py hybrid/ 2>/dev/null || true

echo "=== Archiving patches and backups ==="
mv -n *patch*.py archive_old/ 2>/dev/null || true
mv -n fix_*.py archive_old/ 2>/dev/null || true
mv -n *.bak* archive_old/ 2>/dev/null || true

echo "=== Done ==="