from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


def get_default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True)
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor


class ArtinPolicyNet(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.observation_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.pi = nn.Linear(128, self.action_dim)
        self.v = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> PolicyOutput:
        x = self.mlp(obs)
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return PolicyOutput(logits=logits, value=value)


def sample_action_and_logp(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    logp = dist.log_prob(action)
    entropy = dist.entropy()
    return action, logp, entropy

