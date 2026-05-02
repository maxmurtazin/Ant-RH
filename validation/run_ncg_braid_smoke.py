#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from core.ncg_braid_spectral import (
    BraidNCGConfig,
    enumerate_braid_words,
    build_dirac_operator,
    full_ncg_loss,
    load_zeros,
    safe_eigvalsh,
    spectral_zeta_from_eigs,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test: NCG braid spectral layer for Ant-RH")
    ap.add_argument("--n_strands", type=int, default=4)
    ap.add_argument("--max_word_len", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--zeros", default=None, help="Optional txt/csv/json with zeta zero ordinates")
    ap.add_argument("--out_dir", default="runs/ncg_braid_smoke")
    args = ap.parse_args()

    cfg = BraidNCGConfig(
        n_strands=args.n_strands,
        max_word_len=args.max_word_len,
        dim=args.dim,
        device=args.device,
        dtype=args.dtype,
    )
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    words = enumerate_braid_words(cfg.n_strands, cfg.max_word_len, cfg.dim)
    zeros = load_zeros(args.zeros, cfg, count=max(128, cfg.dim))
    D = build_dirac_operator(words, cfg)
    loss, stats = full_ncg_loss(D, zeros, cfg)
    eigs = safe_eigvalsh(D)

    report = {
        "version": "NCG-Braid Spectral Smoke V0.1",
        "n_words": len(words),
        "config": cfg.__dict__ | {"torch_dtype": str(cfg.torch_dtype)},
        "loss_stats": stats,
        "zeta_D_s2": float(spectral_zeta_from_eigs(eigs, 2.0).detach().cpu()),
        "first_eigs": [float(x) for x in eigs[:20].detach().cpu()],
        "first_words": [" ".join([f"s{i}{'-' if s < 0 else '+'}" for i, s in w]) or "e" for w in words[:20]],
    }
    (out / "ncg_braid_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"D": D.detach().cpu(), "eigs": eigs.detach().cpu()}, out / "operator.pt")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
