#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.braid_tokenizer import BraidTokenizer
from core.dtes_braid_dsl import serialize_episode, serialize_from_aco_logs
from core.topological_llm import TopologicalTinyLM


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return bool(default)
    return str(x).lower() in {"1", "true", "yes", "y"}


def _is_valid_word(a_list: Any, *, min_len: int = 3, max_len: int = 8) -> bool:
    if not isinstance(a_list, list):
        return False
    if len(a_list) < int(min_len) or len(a_list) > int(max_len):
        return False
    for a in a_list:
        try:
            ai = int(a)
        except Exception:
            return False
        if ai == 0:
            return False
    return True


def _near_dup_key(word: List[int]) -> tuple:
    # coarse key for near-duplicate detection: signs + capped magnitudes
    w = [int(a) for a in word]
    mags = [min(abs(a), 6) for a in w]
    signs = [1 if a > 0 else -1 for a in w]
    return (len(w), tuple(signs), tuple(mags))


def _is_near_duplicate(word: List[int], kept: List[List[int]]) -> bool:
    """
    Treat as near-duplicate if it matches an existing word exactly,
    or differs in <=1 position for same-length words (Hamming distance).
    """
    w = [int(a) for a in word]
    wt = tuple(w)
    for k in kept:
        kt = tuple(int(a) for a in k)
        if wt == kt:
            return True
        if len(wt) == len(kt) and len(wt) <= 8:
            diffs = sum(1 for a, b in zip(wt, kt) if a != b)
            if diffs <= 1:
                return True
    return False


def _word_to_episode(word: List[int], *, loss: float, idx: int, tag: str) -> Dict[str, Any]:
    w = [int(a) for a in word]
    events = [{"i": int(abs(a)), "dir": 1 if int(a) > 0 else -1} for a in w]
    n_strands = max([ev["i"] for ev in events] + [1]) + 1
    spec_err = float(loss)
    reward = -spec_err
    return {
        "id": f"{tag}_{idx}",
        "state": {"t": idx, "energy": spec_err, "cluster": idx % 8},
        "braid": {"n_strands": int(n_strands), "events": events},
        "operator": {"type": "artin_aco_candidate"},
        "spectrum": {"error": spec_err, "spacing_error": 0.0, "self_adjoint_error": 0.0},
        "reward": {"value": float(reward)},
    }


def _load_aco_quality_words(
    root: Path,
    *,
    max_train_loss: float,
    dedup_near: bool,
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts: {"word": [...], "loss": float, "source": str}
    Reads runs/artin_aco_best.json and (optionally) an ACO bank file if present.
    """
    out: List[Dict[str, Any]] = []
    best_path = root / "runs/artin_aco_best.json"
    if best_path.exists():
        try:
            best = json.loads(best_path.read_text(encoding="utf-8"))
            best_loss = _to_float(best.get("best_loss", float("inf")), default=float("inf"))
            best_words = best.get("best_words", [])
            if isinstance(best_words, list):
                for w in best_words:
                    if not isinstance(w, list):
                        continue
                    out.append({"word": [int(a) for a in w], "loss": float(best_loss), "source": "aco_best"})
        except Exception:
            pass

    # Optional bank files (if present in future runs)
    bank_candidates: List[Path] = []
    for name in ("artin_aco_bank.json", "artin_aco_bank.jsonl", "artin_aco_candidates.json", "artin_aco_candidates.jsonl"):
        p = root / "runs" / name
        if p.exists():
            bank_candidates.append(p)

    for p in bank_candidates:
        try:
            if p.suffix == ".jsonl":
                for line in p.read_text(encoding="utf-8").splitlines():
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    w = obj.get("a_list", obj.get("word", None))
                    loss = _to_float(obj.get("loss", obj.get("best_loss", float("inf"))), default=float("inf"))
                    if isinstance(w, list):
                        out.append({"word": [int(a) for a in w], "loss": float(loss), "source": p.name})
            else:
                obj = json.loads(p.read_text(encoding="utf-8"))
                items = obj if isinstance(obj, list) else obj.get("candidates", [])
                if isinstance(items, list):
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        w = it.get("a_list", it.get("word", None))
                        loss = _to_float(it.get("loss", it.get("best_loss", float("inf"))), default=float("inf"))
                        if isinstance(w, list):
                            out.append({"word": [int(a) for a in w], "loss": float(loss), "source": p.name})
        except Exception:
            pass

    # Filter + dedup
    filtered: List[Dict[str, Any]] = []
    kept_words: List[List[int]] = []
    seen_near_keys = set()
    for item in out:
        w = item.get("word", None)
        if not isinstance(w, list):
            continue
        loss = _to_float(item.get("loss", float("inf")), default=float("inf"))
        if np.isfinite(loss) and float(loss) >= float(max_train_loss):
            continue
        if not _is_valid_word(w, min_len=3, max_len=8):
            continue
        if bool(dedup_near):
            nk = _near_dup_key(w)
            if nk in seen_near_keys:
                continue
            if _is_near_duplicate(w, kept_words):
                continue
            seen_near_keys.add(nk)
        kept_words.append([int(a) for a in w])
        filtered.append({"word": [int(a) for a in w], "loss": float(loss), "source": str(item.get("source", "aco"))})
    return filtered


def _synthetic_episode(idx: int, max_sigma: int = 8) -> Dict[str, Any]:
    length_choices = [3, 4, 5, 6, 7, 8, 9, 10]
    length_weights = [1, 2, 3, 3, 3, 2, 2, 1]
    L = random.choices(length_choices, weights=length_weights, k=1)[0]
    events = []
    base_directions = [1 if (idx + j) % 2 == 0 else -1 for j in range(L)]
    if idx % 3 == 0:
        base_directions = list(reversed(base_directions))
    for j in range(L):
        sigma = random.randint(1, max_sigma)
        direction = base_directions[j]
        if j > 0 and events[-1]["i"] == sigma and events[-1]["dir"] == direction:
            sigma = (sigma % max_sigma) + 1
            direction = -direction
        events.append({"i": sigma, "dir": direction})
    sigma_values = [ev["i"] for ev in events]
    if sigma_values[:3] == [4, 2, 1] and [ev["dir"] for ev in events[:3]] == [1, -1, -1]:
        events[0]["i"] = 3
        events[-1]["dir"] *= -1
    spec_err = random.random() * 10.0
    spacing_err = random.random()
    balance_bonus = 0.25 if abs(sum(ev["dir"] for ev in events)) <= 1 else 0.0
    diversity_bonus = len(set((ev["i"], ev["dir"]) for ev in events)) / max(len(events), 1)
    reward = -spec_err - spacing_err + random.random() + balance_bonus + 0.2 * diversity_bonus
    return {
        "id": f"synthetic_{idx}",
        "state": {"t": idx, "energy": spec_err, "cluster": idx % 8},
        "braid": {"n_strands": max(ev["i"] for ev in events) + 1, "events": events},
        "operator": {"type": "braid_laplacian"},
        "spectrum": {"error": spec_err, "spacing_error": spacing_err, "self_adjoint_error": 0.0},
        "reward": {"value": reward},
    }


def build_dataset_records(
    root: Path,
    *,
    quality_filter: bool = True,
    max_train_loss: float = 0.02,
    dedup_near: bool = True,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if bool(quality_filter):
        quality = _load_aco_quality_words(root, max_train_loss=float(max_train_loss), dedup_near=bool(dedup_near))
        if quality:
            # Oversample high-quality episodes x5
            episodes: List[Dict[str, Any]] = []
            for rep in range(5):
                for idx, it in enumerate(quality):
                    ep_idx = rep * len(quality) + idx
                    ep = _word_to_episode(it["word"], loss=float(it["loss"]), idx=ep_idx, tag="aco_quality")
                    episodes.append(ep)
            random.shuffle(episodes)
            for ep in episodes:
                records.append({"dsl": serialize_episode(ep), "source": "aco_quality"})
            return records

    for dsl in serialize_from_aco_logs(
        aco_history_path=str(root / "runs/artin_aco_history.csv"),
        best_path=str(root / "runs/artin_aco_best.json"),
    ):
        records.append({"dsl": dsl, "source": "aco"})

    journal_path = root / "runs/lab_journal.jsonl"
    if journal_path.exists():
        try:
            with journal_path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    metrics = obj.get("key_metrics", {}) if isinstance(obj, dict) else {}
                    spec_err = _to_float(metrics.get("operator_spectral_loss", idx + 1.0))
                    reward = -spec_err
                    record = {
                        "id": f"journal_{idx}",
                        "state": {"t": idx, "energy": spec_err, "cluster": idx % 8},
                        "braid": {"n_strands": 4, "events": [{"i": 1, "dir": 1}, {"i": 2, "dir": -1}, {"i": 1, "dir": 1}]},
                        "operator": {"type": "braid_laplacian"},
                        "spectrum": {
                            "error": spec_err,
                            "spacing_error": _to_float(metrics.get("operator_spacing_loss", 0.0)),
                            "self_adjoint_error": 0.0,
                        },
                        "reward": {"value": reward},
                    }
                    records.append({"dsl": serialize_episode(record), "source": "journal"})
        except Exception:
            pass

    while len(records) < 128:
        idx = len(records)
        records.append({"dsl": serialize_episode(_synthetic_episode(idx)), "source": "synthetic"})
    return records


class TokenDataset(Dataset):
    def __init__(self, seqs: List[List[int]], seq_len: int, pad_id: int):
        self.seqs = seqs
        self.seq_len = int(seq_len)
        self.pad_id = int(pad_id)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx][: self.seq_len + 1]
        if len(seq) < self.seq_len + 1:
            seq = seq + [self.pad_id] * (self.seq_len + 1 - len(seq))
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


def main() -> None:
    ap = argparse.ArgumentParser(description="Train topological tiny LM")
    ap.add_argument("--dataset", type=str, default="runs/topological_lm_dataset.jsonl")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default="runs/topological_lm")
    ap.add_argument("--build_dataset_only", type=str, default="False")
    ap.add_argument("--quality_filter", type=str, default="True")
    ap.add_argument("--max_train_loss", type=float, default=0.02)
    ap.add_argument("--dedup_near", type=str, default="True")
    args = ap.parse_args()

    root = Path(ROOT)
    dataset_path = root / str(args.dataset)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    records = build_dataset_records(
        root,
        quality_filter=_to_bool(args.quality_filter, default=True),
        max_train_loss=float(args.max_train_loss),
        dedup_near=_to_bool(args.dedup_near, default=True),
    )
    with dataset_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if str(args.build_dataset_only).lower() in {"1", "true", "yes", "y"}:
        return

    dsl_texts = [r["dsl"] for r in records]
    tokenizer = BraidTokenizer(max_sigma=16, n_bins=32)
    tokenizer.fit(dsl_texts)
    seqs = [tokenizer.encode(text) for text in dsl_texts]

    out_dir = root / str(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")

    pad_id = tokenizer.vocab["PAD"]
    dataset = TokenDataset(seqs, seq_len=int(args.seq_len), pad_id=pad_id)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True)

    device = _device()
    model = TopologicalTinyLM(vocab_size=len(tokenizer.vocab), max_seq_len=int(args.seq_len)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    history_rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: List[float] = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=pad_id)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = sum(losses) / max(1, len(losses))
        perplexity = float(torch.exp(torch.tensor(mean_loss)).item())
        history_rows.append({"epoch": epoch, "train_loss": mean_loss, "perplexity": perplexity})

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "vocab_size": len(tokenizer.vocab),
                "d_model": model.d_model,
                "n_heads": model.n_heads,
                "n_layers": model.n_layers,
                "max_seq_len": model.max_seq_len,
            },
        },
        out_dir / "model.pt",
    )

    with (out_dir / "train_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "perplexity"])
        for row in history_rows:
            writer.writerow([row["epoch"], row["train_loss"], row["perplexity"]])

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": str(dataset_path),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "seq_len": int(args.seq_len),
                "device": str(device),
                "num_records": len(records),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
