#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from typing import List

import torch
import torch.nn as nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TopologicalTinyLM(nn.Module):
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
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.max_seq_len = int(max_seq_len)
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=self.n_layers)
        self.ln = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        mask = torch.triu(torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1)
        x = self.decoder(x, mask=mask)
        x = self.ln(x)
        return self.head(x)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    context_ids,
    max_new_tokens: int = 64,
    temperature: float = 1.1,
    top_k: int = 40,
    top_p: float = 0.9,
    no_repeat_ngram_size: int = 3,
    repetition_penalty: float = 1.2,
) -> List[int]:
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
        if not banned:
            return logits
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

    model.eval()
    device = next(model.parameters()).device
    ids = list(int(x) for x in context_ids)
    eos_id = tokenizer.vocab.get("EOS", None)
    for _ in range(int(max_new_tokens)):
        x = torch.tensor([ids[-model.max_seq_len :]], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        if float(temperature) <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = logits / max(float(temperature), 1e-6)
            recent_window = max(6, int(no_repeat_ngram_size) * 4)
            logits = _apply_repetition_penalty(logits, ids[-recent_window:], float(repetition_penalty))
            logits = _apply_no_repeat_ngram(logits, ids, int(no_repeat_ngram_size))
            if int(top_k) > 0:
                values, indices = torch.topk(logits, k=min(int(top_k), logits.shape[-1]))
                next_local = _sample_top_p(values, float(top_p))
                next_id = int(indices[next_local].item())
            else:
                next_id = _sample_top_p(logits, float(top_p))
        ids.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
    return ids
