from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _first_present(data: Dict[str, Any], keys):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _load_zeta_zeros(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("input JSON must contain an object")
    zeta = _first_present(data, ("zeta_zeros", "true_zeros", "known_zeros"))
    if zeta is None:
        raise ValueError("input JSON missing zeta_zeros")
    zeta = np.asarray(zeta, dtype=float).reshape(-1)
    zeta = np.sort(np.abs(zeta[np.isfinite(zeta)]))
    zeta = zeta[zeta > 0.0]
    if zeta.size == 0:
        raise ValueError("zeta_zeros must contain positive finite values")
    return zeta


def _load_eigenvalues(path: Path):
    try:
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            eig = arr[:, 1]
        else:
            eig = np.asarray(arr, dtype=float).reshape(-1)
    except ValueError:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                rows.append(float(row[-1]))
        eig = np.asarray(rows, dtype=float)

    eig = np.asarray(eig, dtype=float).reshape(-1)
    eig = eig[np.isfinite(eig)]
    if eig.size == 0:
        raise ValueError("eigenvalues file contains no finite values")
    return np.sort(eig)


def _spacing(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.asarray([], dtype=float)
    return values[1:] - values[:-1]


def _scaled_mse(eig, zeta):
    if eig.size < 2 or zeta.size < 2:
        return None
    er = eig[-1] - eig[0]
    zr = zeta[-1] - zeta[0]
    if abs(er) < 1e-12 or abs(zr) < 1e-12:
        return None
    e_scaled = (eig - eig[0]) / er
    z_scaled = (zeta - zeta[0]) / zr
    return float(np.mean((e_scaled - z_scaled) ** 2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate V11 Artin DTES spectrum.")
    parser.add_argument("--eigenvalues", default="runs/v11_artin_dtes/eigenvalues.csv")
    parser.add_argument("--input", default="runs/dtes_spectral_input.json")
    parser.add_argument("--out", default="runs/v11_artin_dtes/artin_validation_report.json")
    parser.add_argument("--k", type=int, default=50)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    eig = _load_eigenvalues(Path(args.eigenvalues))
    zeta = _load_zeta_zeros(Path(args.input))

    k = min(int(args.k), eig.size, zeta.size)
    if k <= 0:
        raise ValueError("no overlapping eigenvalues/zeta zeros to validate")

    eig_k = eig[:k]
    zeta_k = zeta[:k]
    de = _spacing(eig_k)
    dz = _spacing(zeta_k)

    raw_mse = float(np.mean((eig_k - zeta_k) ** 2))
    spacing_mse = float(np.mean((de - dz) ** 2)) if de.size and dz.size else None
    eig_range = float(eig_k.max() - eig_k.min())
    zeta_range = float(zeta_k.max() - zeta_k.min())
    range_ratio = float(eig_range / (zeta_range + 1e-12))

    report = {
        "version": "V11 Artin trace validation",
        "eigenvalues": str(args.eigenvalues),
        "input": str(args.input),
        "k": int(k),
        "raw_mse": raw_mse,
        "spacing_mse": spacing_mse,
        "eig_range": eig_range,
        "zeta_range": zeta_range,
        "range_ratio": range_ratio,
        "scaled_mse": _scaled_mse(eig_k, zeta_k),
        "finite_outputs": bool(np.all(np.isfinite(eig_k))),
        "note": "Validation for trainable Selberg-Artin DTES system; not proof of RH.",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
