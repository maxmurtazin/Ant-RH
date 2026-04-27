from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.operator_health import validate_operator_health


def load_operator(path):
    path = str(path)
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu")
    arr = np.load(path)
    return torch.tensor(arr)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Validate V10.10 self-adjoint operator health."
    )
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--project", action="store_true")
    p.add_argument("--self-adjoint-tol", type=float, default=1e-8)
    p.add_argument("--jitter-eps", type=float, default=1e-8)
    p.add_argument("--save-eigenvalues")
    args = p.parse_args(argv)

    H = load_operator(args.input)

    H_proj, eig, report = validate_operator_health(
        H,
        self_adjoint_tol=args.self_adjoint_tol,
        strict_self_adjoint=False,
        jitter_eps=args.jitter_eps,
    )
    report["project_requested"] = bool(args.project)
    report["input"] = str(args.input)
    report["note"] = "operator-level Hilbert–Pólya compatibility check; not a proof of RH"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.save_eigenvalues and eig is not None:
        eig_path = Path(args.save_eigenvalues)
        eig_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(eig_path, eig.detach().cpu().numpy(), delimiter=",")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
