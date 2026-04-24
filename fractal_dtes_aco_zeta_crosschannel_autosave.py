#!/usr/bin/env python3

import numpy as np
import mpmath as mp
import json
import argparse

mp.mp.dps = 50


# ----------------------------
# ZETA
# ----------------------------
def zeta_on_critical_line(t):
    s = 0.5 + 1j * t
    return mp.zeta(s)


# ----------------------------
# SIMPLE DTES-ACO MOCK (твоя логика сюда)
# ----------------------------
def find_candidates(t_min, t_max, n0):
    """
    Упрощённый поиск кандидатов (замени на свой ACO/DTES)
    """
    ts = np.linspace(t_min, t_max, n0)
    vals = [abs(zeta_on_critical_line(t)) for t in ts]

    candidates = []
    for i in range(1, len(ts) - 1):
        if vals[i] < vals[i - 1] and vals[i] < vals[i + 1]:
            candidates.append(ts[i])

    return candidates


# ----------------------------
# AUTO SAVE
# ----------------------------
def save_candidates(candidates, filename="dtes_candidates.json"):
    data = {
        "candidates": [
            {"rank": i + 1, "t": float(t)}
            for i, t in enumerate(candidates)
        ],
        "count": len(candidates),
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[SAVE] {filename} ({len(candidates)} candidates)")


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--t_min", type=float, required=True)
    parser.add_argument("--t_max", type=float, required=True)
    parser.add_argument("--n0", type=int, default=2000)
    parser.add_argument("--output", default="dtes_candidates.json")

    args = parser.parse_args()

    print("[RUN] DTES candidate search")

    candidates = find_candidates(
        args.t_min,
        args.t_max,
        args.n0
    )

    print("\nCandidates:")
    for t in candidates[:20]:
        print(f"t = {t:.12f}")

    if len(candidates) > 20:
        print(f"... ({len(candidates)} total)")

    # 🔥 AUTO SAVE
    save_candidates(candidates, args.output)


if __name__ == "__main__":
    main()