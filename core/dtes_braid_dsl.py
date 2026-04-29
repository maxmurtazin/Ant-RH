#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _attr_value(text: str, key: str, default: str = "") -> str:
    m = re.search(rf'{re.escape(key)}="([^"]*)"', text)
    return m.group(1) if m else default


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def serialize_episode(record: dict) -> str:
    episode_id = str(record.get("id", "unknown"))
    state = record.get("state", {}) or {}
    braid = record.get("braid", {}) or {}
    operator = record.get("operator", {}) or {}
    spectrum = record.get("spectrum", {}) or {}
    reward = record.get("reward", {}) or {}

    sigma_lines: List[str] = []
    for ev in braid.get("events", []) or []:
        try:
            idx = int(abs(int(ev.get("i", 0))))
            direction = int(ev.get("dir", 1))
        except Exception:
            continue
        if idx <= 0 or direction == 0:
            continue
        sigma_lines.append(f'<sigma i="{idx}" dir="{1 if direction > 0 else -1}" />')

    return "\n".join(
        [
            f'<episode id="{episode_id}">',
            f'<state t="{int(state.get("t", 0))}" energy="{_to_float(state.get("energy", 0.0)):.6g}" cluster="{int(state.get("cluster", 0))}" />',
            f'<braid n_strands="{int(braid.get("n_strands", 0))}">',
            *sigma_lines,
            "</braid>",
            f'<operator type="{str(operator.get("type", "braid_laplacian"))}" />',
            (
                f'<spectrum error="{_to_float(spectrum.get("error", 0.0)):.6g}" '
                f'spacing_error="{_to_float(spectrum.get("spacing_error", 0.0)):.6g}" '
                f'self_adjoint_error="{_to_float(spectrum.get("self_adjoint_error", 0.0)):.6g}" />'
            ),
            f'<reward value="{_to_float(reward.get("value", 0.0)):.6g}" />',
            "</episode>",
        ]
    )


def parse_episode(dsl: str) -> dict:
    text = str(dsl or "")
    episode_line = re.search(r"<episode[^>]*>", text)
    state_line = re.search(r"<state[^>]*/>", text)
    braid_line = re.search(r"<braid[^>]*>", text)
    operator_line = re.search(r"<operator[^>]*/>", text)
    spectrum_line = re.search(r"<spectrum[^>]*/>", text)
    reward_line = re.search(r"<reward[^>]*/>", text)
    sigma_lines = re.findall(r"<sigma[^>]*/>", text)

    return {
        "id": _attr_value(episode_line.group(0), "id", "unknown") if episode_line else "unknown",
        "state": {
            "t": int(_to_float(_attr_value(state_line.group(0), "t", "0"))) if state_line else 0,
            "energy": _to_float(_attr_value(state_line.group(0), "energy", "0")) if state_line else 0.0,
            "cluster": int(_to_float(_attr_value(state_line.group(0), "cluster", "0"))) if state_line else 0,
        },
        "braid": {
            "n_strands": int(_to_float(_attr_value(braid_line.group(0), "n_strands", "0"))) if braid_line else 0,
            "events": [
                {
                    "i": int(_to_float(_attr_value(sig, "i", "0"))),
                    "dir": int(_to_float(_attr_value(sig, "dir", "1"))),
                }
                for sig in sigma_lines
                if int(_to_float(_attr_value(sig, "i", "0"))) > 0 and int(_to_float(_attr_value(sig, "dir", "1"))) != 0
            ],
        },
        "operator": {"type": _attr_value(operator_line.group(0), "type", "braid_laplacian") if operator_line else "braid_laplacian"},
        "spectrum": {
            "error": _to_float(_attr_value(spectrum_line.group(0), "error", "0")) if spectrum_line else 0.0,
            "spacing_error": _to_float(_attr_value(spectrum_line.group(0), "spacing_error", "0")) if spectrum_line else 0.0,
            "self_adjoint_error": _to_float(_attr_value(spectrum_line.group(0), "self_adjoint_error", "0")) if spectrum_line else 0.0,
        },
        "reward": {"value": _to_float(_attr_value(reward_line.group(0), "value", "0")) if reward_line else 0.0},
    }


def _word_to_sigma_events(word: List[int]) -> List[Dict[str, int]]:
    events: List[Dict[str, int]] = []
    for a in word:
        try:
            ai = int(a)
        except Exception:
            continue
        if ai == 0:
            continue
        events.append({"i": abs(ai), "dir": 1 if ai > 0 else -1})
    return events


def serialize_from_aco_logs(
    aco_history_path: str = "runs/artin_aco_history.csv",
    best_path: str = "runs/artin_aco_best.json",
) -> list[str]:
    episodes: List[str] = []
    history_rows: List[Dict[str, Any]] = []
    best_payload: Dict[str, Any] = {}

    try:
        with open(aco_history_path, "r", encoding="utf-8", newline="") as f:
            history_rows = list(csv.DictReader(f))
    except Exception:
        history_rows = []

    try:
        with open(best_path, "r", encoding="utf-8") as f:
            best_payload = json.load(f)
    except Exception:
        best_payload = {}

    best_words = best_payload.get("best_words", []) or []
    word = best_words[0] if best_words else [1, -1, 2]
    sigma_events = _word_to_sigma_events(word)
    base_energy = _to_float(best_payload.get("best_loss", 0.0))

    for idx, row in enumerate(history_rows):
        best_loss = _to_float(row.get("best_loss", base_energy), base_energy)
        mean_loss = _to_float(row.get("mean_loss", best_loss), best_loss)
        reward = -best_loss
        record = {
            "id": f"aco_{idx}",
            "state": {"t": idx, "energy": best_loss, "cluster": idx % 8},
            "braid": {"n_strands": max([ev["i"] for ev in sigma_events], default=2) + 1, "events": sigma_events},
            "operator": {"type": "braid_laplacian"},
            "spectrum": {
                "error": best_loss,
                "spacing_error": abs(mean_loss - best_loss),
                "self_adjoint_error": 0.0,
            },
            "reward": {"value": reward},
        }
        episodes.append(serialize_episode(record))

    if not episodes:
        fallback_record = {
            "id": "aco_fallback",
            "state": {"t": 0, "energy": base_energy, "cluster": 0},
            "braid": {"n_strands": max([ev["i"] for ev in sigma_events], default=2) + 1, "events": sigma_events},
            "operator": {"type": "braid_laplacian"},
            "spectrum": {"error": base_energy, "spacing_error": 0.0, "self_adjoint_error": 0.0},
            "reward": {"value": -base_energy},
        }
        episodes.append(serialize_episode(fallback_record))

    return episodes
