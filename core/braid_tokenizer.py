#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class BraidTokenizer:
    def __init__(self, max_sigma: int = 16, n_bins: int = 32):
        self.max_sigma = int(max_sigma)
        self.n_bins = int(n_bins)
        self.special_tokens = ["PAD", "BOS", "EOS", "UNK"]
        self.fixed_tokens = [
            "<episode>",
            "</episode>",
            "<state>",
            "<braid>",
            "</braid>",
            "<operator>",
            "<spectrum>",
            "<reward>",
        ]
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self.energy_min = 0.0
        self.energy_max = 1.0
        self.reward_min = -1.0
        self.reward_max = 1.0
        self.spec_min = 0.0
        self.spec_max = 1.0
        self.cluster_max = 16
        self._build_base_vocab()

    def _build_base_vocab(self) -> None:
        tokens: List[str] = []
        tokens.extend(self.special_tokens)
        tokens.extend(self.fixed_tokens)
        for i in range(1, self.max_sigma + 1):
            tokens.append(f"SIGMA_{i}_PLUS")
            tokens.append(f"SIGMA_{i}_MINUS")
        for i in range(self.n_bins):
            tokens.append(f"ENERGY_BIN_{i}")
            tokens.append(f"CLUSTER_{i}")
            tokens.append(f"REWARD_BIN_{i}")
            tokens.append(f"SPECTRAL_ERROR_BIN_{i}")
        self.vocab = {tok: idx for idx, tok in enumerate(tokens)}
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def _bin_value(self, value: float, lo: float, hi: float) -> int:
        if not math.isfinite(value):
            return 0
        hi = max(hi, lo + 1e-9)
        x = (float(value) - lo) / (hi - lo)
        idx = int(max(0, min(self.n_bins - 1, math.floor(x * self.n_bins))))
        return idx

    def fit(self, dsl_texts: list[str]) -> None:
        energies: List[float] = []
        rewards: List[float] = []
        specs: List[float] = []
        clusters: List[int] = []
        for text in dsl_texts:
            for val in re.findall(r'energy="([^"]+)"', str(text)):
                try:
                    energies.append(float(val))
                except Exception:
                    pass
            for val in re.findall(r'value="([^"]+)"', str(text)):
                try:
                    rewards.append(float(val))
                except Exception:
                    pass
            for val in re.findall(r'error="([^"]+)"', str(text)):
                try:
                    specs.append(float(val))
                except Exception:
                    pass
            for val in re.findall(r'cluster="([^"]+)"', str(text)):
                try:
                    clusters.append(int(float(val)))
                except Exception:
                    pass
        if energies:
            self.energy_min = float(min(energies))
            self.energy_max = float(max(energies))
        if rewards:
            self.reward_min = float(min(rewards))
            self.reward_max = float(max(rewards))
        if specs:
            self.spec_min = float(min(specs))
            self.spec_max = float(max(specs))
        if clusters:
            self.cluster_max = max(max(clusters) + 1, self.n_bins)

    def encode(self, dsl: str) -> list[int]:
        text = str(dsl or "")
        tokens: List[str] = ["BOS"]
        if "<episode" in text:
            tokens.append("<episode>")
        if "<state" in text:
            tokens.append("<state>")
            m_energy = re.search(r'energy="([^"]+)"', text)
            m_cluster = re.search(r'cluster="([^"]+)"', text)
            energy = float(m_energy.group(1)) if m_energy else 0.0
            cluster = int(float(m_cluster.group(1))) if m_cluster else 0
            tokens.append(f"ENERGY_BIN_{self._bin_value(energy, self.energy_min, self.energy_max)}")
            tokens.append(f"CLUSTER_{max(0, min(self.n_bins - 1, cluster))}")
        if "<braid" in text:
            tokens.append("<braid>")
            for i_str, dir_str in re.findall(r'<sigma[^>]*i="([^"]+)"[^>]*dir="([^"]+)"', text):
                try:
                    idx = max(1, min(self.max_sigma, int(float(i_str))))
                    direction = int(float(dir_str))
                except Exception:
                    continue
                sign = "PLUS" if direction > 0 else "MINUS"
                tokens.append(f"SIGMA_{idx}_{sign}")
            tokens.append("</braid>")
        if "<operator" in text:
            tokens.append("<operator>")
        if "<spectrum" in text:
            tokens.append("<spectrum>")
            m_spec = re.search(r'error="([^"]+)"', text)
            spec = float(m_spec.group(1)) if m_spec else 0.0
            tokens.append(f"SPECTRAL_ERROR_BIN_{self._bin_value(spec, self.spec_min, self.spec_max)}")
        if "<reward" in text:
            tokens.append("<reward>")
            m_reward = re.search(r'value="([^"]+)"', text)
            reward = float(m_reward.group(1)) if m_reward else 0.0
            tokens.append(f"REWARD_BIN_{self._bin_value(reward, self.reward_min, self.reward_max)}")
        if "</episode>" in text:
            tokens.append("</episode>")
        tokens.append("EOS")
        unk = self.vocab["UNK"]
        return [self.vocab.get(tok, unk) for tok in tokens]

    def decode(self, ids: list[int]) -> str:
        toks = [self.inv_vocab.get(int(i), "UNK") for i in ids]
        return " ".join(toks)

    def save(self, path) -> None:
        payload = {
            "max_sigma": self.max_sigma,
            "n_bins": self.n_bins,
            "vocab": self.vocab,
            "energy_min": self.energy_min,
            "energy_max": self.energy_max,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "spec_min": self.spec_min,
            "spec_max": self.spec_max,
            "cluster_max": self.cluster_max,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls(max_sigma=int(payload["max_sigma"]), n_bins=int(payload["n_bins"]))
        tok.vocab = {str(k): int(v) for k, v in payload["vocab"].items()}
        tok.inv_vocab = {int(v): str(k) for k, v in tok.vocab.items()}
        tok.energy_min = float(payload.get("energy_min", 0.0))
        tok.energy_max = float(payload.get("energy_max", 1.0))
        tok.reward_min = float(payload.get("reward_min", -1.0))
        tok.reward_max = float(payload.get("reward_max", 1.0))
        tok.spec_min = float(payload.get("spec_min", 0.0))
        tok.spec_max = float(payload.get("spec_max", 1.0))
        tok.cluster_max = int(payload.get("cluster_max", tok.n_bins))
        return tok
