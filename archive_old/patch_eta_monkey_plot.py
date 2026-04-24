#!/usr/bin/env python3
"""
patch_eta_monkey_plot.py

Robust fix for:
  AttributeError: 'ETAFractalDTESACOZeta' object has no attribute '_plot_all_local'

This patch does NOT try to insert the method inside the class.
Instead it injects a function before `if __name__ == "__main__":`
and assigns:

  ETAFractalDTESACOZeta._plot_all_local = _eta_plot_all_local

Usage:
  cd /Users/machome/Project/fractal_dtes_aco_zeta_figures_pack
  python3 patch_eta_monkey_plot.py
  python3 fractal_dtes_aco_zeta_eta.py
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("fractal_dtes_aco_zeta_eta.py")

PATCH_BLOCK = r