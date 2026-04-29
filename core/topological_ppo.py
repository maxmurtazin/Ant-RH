#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.braid_tokenizer import BraidTokenizer
from core.dtes_braid_executor import braid_tokens_to_artin_word
from core.topological_llm import TopologicalTinyLM


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TopologicalPPOPolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base = TopologicalTinyLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.value_head = nn.Linear(self.base.d_model, 1)

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        positions = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.base.token_emb(input_ids) + self.base.pos_emb(positions)
        mask = torch.triu(torch.ones(seqlen, seqlen, device=input_ids.device, dtype=torch.bool), diagonal=1)
        x = self.base.decoder(x, mask=mask)
        x = self.base.ln(x)
        return x

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden_states(input_ids)
        logits = self.base.head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


def build_prefix_ids(tokenizer: BraidTokenizer) -> List[int]:
    bos = tokenizer.vocab["BOS"]
    prefix = [
        bos,
        tokenizer.vocab.get("<episode>", bos),
        tokenizer.vocab.get("<state>", bos),
        tokenizer.vocab.get("ENERGY_BIN_0", bos),
        tokenizer.vocab.get("CLUSTER_0", bos),
        tokenizer.vocab.get("<spectrum>", bos),
        tokenizer.vocab.get("SPECTRAL_ERROR_BIN_0", bos),
        tokenizer.vocab.get("<reward>", bos),
        tokenizer.vocab.get(f"REWARD_BIN_{max(0, tokenizer.n_bins - 1)}", bos),
        tokenizer.vocab.get("<braid>", bos),
    ]
    return [int(x) for x in prefix]


def _apply_repetition_penalty(logits: torch.Tensor, recent_ids: List[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or not recent_ids:
        return logits
    adjusted = logits.clone()
    for token_id in set(int(x) for x in recent_ids):
        value = adjusted[token_id]
        adjusted[token_id] = value / penalty if float(value) > 0 else value * penalty
    return adjusted


def _apply_no_repeat_ngram(logits: torch.Tensor, ids: List[int], ngram_size: int) -> torch.Tensor:
    if ngram_size <= 1 or len(ids) < ngram_size - 1:
        return logits
    prefix = tuple(ids[-(ngram_size - 1) :])
    banned = set()
    for i in range(len(ids) - ngram_size + 1):
        if tuple(ids[i : i + ngram_size - 1]) == prefix:
            banned.add(int(ids[i + ngram_size - 1]))
    adjusted = logits.clone()
    for token_id in banned:
        adjusted[token_id] = float("-inf")
    return adjusted


def _sample_top_p(logits: torch.Tensor, p: float) -> int:
    probs = torch.softmax(logits, dim=-1)
    if p <= 0.0 or p >= 1.0:
        return int(torch.multinomial(probs, num_samples=1).item())
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative <= p
    keep[0] = True
    filtered_probs = sorted_probs[keep]
    filtered_indices = sorted_indices[keep]
    filtered_probs = filtered_probs / filtered_probs.sum()
    return int(filtered_indices[torch.multinomial(filtered_probs, num_samples=1)].item())


def load_pretrained_lm(model_path: Path, tokenizer: BraidTokenizer, device: torch.device) -> tuple[TopologicalTinyLM, Dict[str, Any]]:
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = TopologicalTinyLM(
        vocab_size=int(cfg.get("vocab_size", len(tokenizer.vocab))),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        n_layers=int(cfg.get("n_layers", 4)),
        max_seq_len=int(cfg.get("max_seq_len", 256)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def load_ppo_policy(path: Path, device: torch.device) -> tuple[TopologicalPPOPolicy, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config", {})
    policy = TopologicalPPOPolicy(
        vocab_size=int(cfg.get("vocab_size", 128)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        n_layers=int(cfg.get("n_layers", 4)),
        max_seq_len=int(cfg.get("max_seq_len", 256)),
    ).to(device)
    policy.base.load_state_dict(ckpt["base_model_state"])
    policy.value_head.load_state_dict(ckpt["value_head_state"])
    policy.eval()
    return policy, cfg


@torch.no_grad()
def sample_policy_episode(
    policy: TopologicalPPOPolicy,
    ref_model: TopologicalTinyLM,
    tokenizer: BraidTokenizer,
    prefix_ids: List[int],
    max_new_tokens: int = 32,
    temperature: float = 1.1,
    top_k: int = 40,
    top_p: float = 0.9,
    no_repeat_ngram_size: int = 3,
    repetition_penalty: float = 1.2,
) -> Dict[str, Any]:
    device = next(policy.parameters()).device
    ids = [int(x) for x in prefix_ids]
    eos_id = tokenizer.vocab.get("EOS")
    braid_end_id = tokenizer.vocab.get("</braid>")
    action_ids: List[int] = []
    old_log_probs: List[float] = []
    ref_log_probs: List[float] = []
    values: List[float] = []
    states: List[List[int]] = []

    for _ in range(int(max_new_tokens)):
        x = torch.tensor([ids[-policy.base.max_seq_len :]], dtype=torch.long, device=device)
        policy_logits, policy_values = policy(x)
        ref_logits = ref_model(x)
        step_logits = policy_logits[0, -1]
        sample_logits = step_logits / max(float(temperature), 1e-6)
        sample_logits = _apply_repetition_penalty(sample_logits, ids[-max(6, no_repeat_ngram_size * 4) :], repetition_penalty)
        sample_logits = _apply_no_repeat_ngram(sample_logits, ids, no_repeat_ngram_size)
        if int(top_k) > 0:
            values_topk, indices_topk = torch.topk(sample_logits, k=min(int(top_k), sample_logits.shape[-1]))
            next_local = _sample_top_p(values_topk, float(top_p))
            next_id = int(indices_topk[next_local].item())
        else:
            next_id = _sample_top_p(sample_logits, float(top_p))

        states.append(list(ids[-policy.base.max_seq_len :]))
        action_ids.append(next_id)
        old_log_probs.append(float(torch.log_softmax(step_logits, dim=-1)[next_id].item()))
        ref_log_probs.append(float(torch.log_softmax(ref_logits[0, -1], dim=-1)[next_id].item()))
        values.append(float(policy_values[0, -1].item()))
        ids.append(next_id)

        if eos_id is not None and next_id == int(eos_id):
            break
        if braid_end_id is not None and next_id == int(braid_end_id):
            break
        token_name = tokenizer.inv_vocab.get(next_id, "UNK")
        if token_name not in {"EOS", "</braid>"} and not token_name.startswith("SIGMA_"):
            break

    decoded = tokenizer.decode(ids)
    word = braid_tokens_to_artin_word(decoded.split())
    kl_mean = float(sum(max(0.0, p - r) for p, r in zip(old_log_probs, ref_log_probs)) / max(1, len(action_ids)))
    return {
        "full_ids": ids,
        "states": states,
        "action_ids": action_ids,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "values": values,
        "decoded": decoded,
        "word": word,
        "kl_mean": kl_mean,
    }
