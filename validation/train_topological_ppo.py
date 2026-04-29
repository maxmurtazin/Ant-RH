#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.braid_tokenizer import BraidTokenizer
from core.dtes_braid_executor import evaluate_braid_candidate
from core.topological_ppo import (
    TopologicalPPOPolicy,
    build_prefix_ids,
    get_device,
    load_pretrained_lm,
    sample_policy_episode,
)


def _clip_reward(x: float) -> float:
    return max(-10.0, min(10.0, float(x)))


def _pad_states(states: List[List[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in states)
    batch = []
    lengths = []
    for s in states:
        seq = list(s)
        lengths.append(len(seq))
        seq = seq + [pad_id] * (max_len - len(seq))
        batch.append(seq)
    return (
        torch.tensor(batch, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


def _take_last(logits: torch.Tensor, values: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx = lengths - 1
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, idx], values[rows, idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="PPO fine-tuning for TopologicalLM")
    ap.add_argument("--model", type=str, default="runs/topological_lm/model.pt")
    ap.add_argument("--tokenizer", type=str, default="runs/topological_lm/tokenizer.json")
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out_dir", type=str, default="runs/topological_ppo")
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--gae_lambda", type=float, default=1.0)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--kl_coef", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    args = ap.parse_args()

    root = Path(ROOT)
    out_dir = root / str(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    tokenizer = BraidTokenizer.load(root / str(args.tokenizer))
    ref_model, cfg = load_pretrained_lm(root / str(args.model), tokenizer, device)
    policy = TopologicalPPOPolicy(
        vocab_size=int(cfg.get("vocab_size", len(tokenizer.vocab))),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        n_layers=int(cfg.get("n_layers", 4)),
        max_seq_len=int(cfg.get("max_seq_len", 256)),
    ).to(device)
    policy.base.load_state_dict(ref_model.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(args.lr))
    prefix_ids = build_prefix_ids(tokenizer)
    pad_id = tokenizer.vocab["PAD"]

    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    history_rows: List[Dict[str, Any]] = []
    best_candidates: List[Dict[str, Any]] = []
    best_reward_seen = float("-inf")

    for update in range(1, int(args.updates) + 1):
        policy.eval()
        rollouts: List[Dict[str, Any]] = []
        batch_seen = set()
        duplicate_count = 0
        for _ in range(int(args.batch_size)):
            sample = sample_policy_episode(
                policy,
                ref_model,
                tokenizer,
                prefix_ids,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                no_repeat_ngram_size=int(args.no_repeat_ngram_size),
                repetition_penalty=float(args.repetition_penalty),
            )
            word = sample.get("word", [])
            if not word:
                continue
            result = evaluate_braid_candidate(word)
            base_reward = _clip_reward(float(result.get("reward", -10.0)))
            word_key = tuple(int(x) for x in word)
            unique_bonus = 0.2 if word_key not in batch_seen else -0.5
            if word_key in batch_seen:
                duplicate_count += 1
            else:
                batch_seen.add(word_key)
            valid_bonus = 0.2 if result.get("valid") else -1.0
            reward = _clip_reward(base_reward + valid_bonus + unique_bonus - float(args.kl_coef) * float(sample["kl_mean"]))
            sample["executor_result"] = result
            sample["terminal_reward"] = reward
            rollouts.append(sample)

            candidate_payload = {
                "update": update,
                "word": word,
                "reward": reward,
                "executor_result": result,
                "decoded": sample.get("decoded", ""),
            }
            best_candidates.append(candidate_payload)
            best_candidates = sorted(best_candidates, key=lambda x: float(x.get("reward", -10.0)), reverse=True)[:50]
            best_reward_seen = max(best_reward_seen, reward)

        if not rollouts:
            continue

        states: List[List[int]] = []
        actions: List[int] = []
        old_log_probs: List[float] = []
        old_values: List[float] = []
        returns: List[float] = []
        reward_values: List[float] = []
        kl_values: List[float] = []
        valid_count = 0

        for item in rollouts:
            reward = float(item["terminal_reward"])
            reward_values.append(reward)
            kl_values.append(float(item["kl_mean"]))
            if item["executor_result"].get("valid"):
                valid_count += 1
            for state_ids, action_id, old_lp, old_v in zip(
                item["states"], item["action_ids"], item["old_log_probs"], item["values"]
            ):
                states.append(state_ids)
                actions.append(int(action_id))
                old_log_probs.append(float(old_lp))
                old_values.append(float(old_v))
                returns.append(reward)

        if not states:
            continue

        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        old_values_t = torch.tensor(old_values, dtype=torch.float32, device=device)
        advantages_t = returns_t - old_values_t
        advantages_t = (advantages_t - advantages_t.mean()) / max(float(advantages_t.std(unbiased=False).item()), 1e-6)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)

        policy.train()
        epoch_policy_losses: List[float] = []
        epoch_value_losses: List[float] = []
        epoch_entropies: List[float] = []

        indices = torch.arange(len(states), device=device)
        for _ in range(2):
            perm = indices[torch.randperm(len(indices))]
            for start in range(0, len(states), max(1, int(args.batch_size))):
                batch_idx = perm[start : start + int(args.batch_size)]
                state_batch = [states[int(i.item())] for i in batch_idx]
                x, lengths = _pad_states(state_batch, pad_id, device)
                logits, values = policy(x)
                step_logits, step_values = _take_last(logits, values, lengths)
                log_probs = torch.log_softmax(step_logits, dim=-1)
                chosen_log_probs = log_probs.gather(1, actions_t[batch_idx].unsqueeze(1)).squeeze(1)
                ratios = torch.exp(chosen_log_probs - old_log_probs_t[batch_idx])
                adv = advantages_t[batch_idx]
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1.0 - float(args.clip_eps), 1.0 + float(args.clip_eps)) * adv
                policy_loss = -torch.mean(torch.minimum(surr1, surr2))
                value_loss = F.mse_loss(step_values, returns_t[batch_idx])
                probs = torch.softmax(step_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                loss = policy_loss + float(args.value_coef) * value_loss - float(args.entropy_coef) * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()

                epoch_policy_losses.append(float(policy_loss.detach().cpu()))
                epoch_value_losses.append(float(value_loss.detach().cpu()))
                epoch_entropies.append(float(entropy.detach().cpu()))

        history_rows.append(
            {
                "update": update,
                "reward_mean": float(sum(reward_values) / len(reward_values)),
                "reward_median": float(statistics.median(reward_values)),
                "reward_max": float(max(reward_values)),
                "valid_ratio": float(valid_count / max(1, len(rollouts))),
                "unique_ratio": float(len(batch_seen) / max(1, len(rollouts))),
                "duplicate_count": int(duplicate_count),
                "policy_loss": float(sum(epoch_policy_losses) / max(1, len(epoch_policy_losses))),
                "value_loss": float(sum(epoch_value_losses) / max(1, len(epoch_value_losses))),
                "entropy": float(sum(epoch_entropies) / max(1, len(epoch_entropies))),
                "kl_mean": float(sum(kl_values) / max(1, len(kl_values))),
            }
        )

    torch.save(
        {
            "base_model_state": policy.base.state_dict(),
            "value_head_state": policy.value_head.state_dict(),
            "config": {
                "vocab_size": policy.base.vocab_size,
                "d_model": policy.base.d_model,
                "n_heads": policy.base.n_heads,
                "n_layers": policy.base.n_layers,
                "max_seq_len": policy.base.max_seq_len,
                "best_reward_seen": best_reward_seen,
            },
        },
        out_dir / "policy.pt",
    )

    with (out_dir / "train_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "update",
                "reward_mean",
                "reward_median",
                "reward_max",
                "valid_ratio",
                "unique_ratio",
                "duplicate_count",
                "policy_loss",
                "value_loss",
                "entropy",
                "kl_mean",
            ]
        )
        for row in history_rows:
            writer.writerow(
                [
                    row["update"],
                    row["reward_mean"],
                    row["reward_median"],
                    row["reward_max"],
                    row["valid_ratio"],
                    row["unique_ratio"],
                    row["duplicate_count"],
                    row["policy_loss"],
                    row["value_loss"],
                    row["entropy"],
                    row["kl_mean"],
                ]
            )

    with (out_dir / "best_candidates.json").open("w", encoding="utf-8") as f:
        json.dump(best_candidates[:20], f, indent=2)

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": str(root / str(args.model)),
                "tokenizer": str(root / str(args.tokenizer)),
                "updates": int(args.updates),
                "batch_size": int(args.batch_size),
                "max_new_tokens": int(args.max_new_tokens),
                "lr": float(args.lr),
                "clip_eps": float(args.clip_eps),
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "entropy_coef": float(args.entropy_coef),
                "value_coef": float(args.value_coef),
                "kl_coef": float(args.kl_coef),
                "device": str(device),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
