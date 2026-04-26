from __future__ import annotations

"""Adaptive loss coefficient controller for DTES geometry learning."""

import math
from dataclasses import dataclass, field


@dataclass
class AdaptiveLossController:
    names: list
    init_weights: dict = field(default_factory=dict)
    min_weight: float = 1e-4
    max_weight: float = 20.0
    lr: float = 0.05
    entropy: float = 0.01
    ema_beta: float = 0.95

    def __post_init__(self):
        self.logits = {}
        self.ema_losses = {}
        self.prev_total = None

        for name in self.names:
            w = self.init_weights.get(name, 1.0)
            w = max(self.min_weight, min(self.max_weight, w))
            self.logits[name] = math.log(w)
            self.ema_losses[name] = None

    def weights(self):
        raw = {k: math.exp(v) for k, v in self.logits.items()}
        total = sum(raw.values()) + 1e-12
        return {k: len(raw) * raw[k] / total for k in raw}

    def update(self, losses: dict):
        """
        losses: raw scalar losses from current step.
        Reward is improvement of EMA-normalized total loss.
        """
        weights = self.weights()

        norm_losses = {}
        for k, val in losses.items():
            val = float(val)
            if self.ema_losses[k] is None:
                self.ema_losses[k] = val
            else:
                self.ema_losses[k] = (
                    self.ema_beta * self.ema_losses[k]
                    + (1.0 - self.ema_beta) * val
                )

            norm_losses[k] = val / (self.ema_losses[k] + 1e-12)

        total = sum(weights[k] * norm_losses[k] for k in norm_losses)

        if self.prev_total is None:
            self.prev_total = total
            return weights

        improvement = self.prev_total - total
        self.prev_total = total

        for k in norm_losses:
            direction = 1.0 if improvement > 0 else -1.0
            pressure = norm_losses[k] - 1.0
            self.logits[k] += self.lr * direction * pressure

            self.logits[k] *= 1.0 - self.entropy

            self.logits[k] = max(
                math.log(self.min_weight),
                min(math.log(self.max_weight), self.logits[k]),
            )

        return self.weights()
