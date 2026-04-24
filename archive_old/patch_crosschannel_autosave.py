#!/usr/bin/env python3
"""
patch_crosschannel_autosave.py

Adds automatic saving of DTES candidates to JSON in
fractal_dtes_aco_zeta_crosschannel.py.

Usage:
    python3 patch_crosschannel_autosave.py fractal_dtes_aco_zeta_crosschannel.py

After patching, every normal run creates:
    dtes_candidates.json

The JSON format is compatible with compare_dtes_vs_truth.py:
[
  {"t": 101.317..., "source": "fractal_dtes_aco_zeta_crosschannel", "rank": 1},
  ...
]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


AUTOSAVE_SNIPPET = r