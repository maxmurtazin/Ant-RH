#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.artin_operator import build_geodesic_kernel, build_laplacian, sample_domain
from core.artin_rl_env import ArtinWordEnv
from core.artin_rl_policy import ArtinPolicyNet, get_default_device, sample_action_and_logp
from validation.selberg_trace_loss import compute_selberg_loss


_DTF = np.float64


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_zeros(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"zeros file not found: {p}")
    z = np.loadtxt(str(p), dtype=_DTF)
    z = np.asarray(z, dtype=_DTF).reshape(-1)
    z = z[np.isfinite(z)]
    return z


def _clip_(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _update_pheromone(
    pheromone: Dict[Tuple[int, int], float],
    words: List[List[int]],
    reward: float,
    rho: float,
    tau_min: float = 1e-6,
    tau_max: float = 1e6,
    q: float = 1.0,
) -> None:
    rho = _clip_(rho, 0.0, 1.0)
    for k, v in list(pheromone.items()):
        pheromone[k] = _clip_((1.0 - rho) * float(v), tau_min, tau_max)
    delta = q * float(reward)
    for a_list in words:
        if len(a_list) < 2:
            continue
        for i in range(len(a_list) - 1):
            key = (int(a_list[i]), int(a_list[i + 1]))
            pheromone[key] = _clip_(pheromone.get(key, 1.0) + delta, tau_min, tau_max)


def _pheromone_logits_bias(
    pheromone: Dict[Tuple[int, int], float],
    prev_a: int,
    actions: np.ndarray,
    pheromone_bias: float,
) -> torch.Tensor:
    if pheromone_bias <= 0.0:
        return torch.zeros((int(actions.size),), dtype=torch.float32)
    taus = np.empty((actions.size,), dtype=np.float64)
    for i, a in enumerate(actions):
        taus[i] = float(pheromone.get((int(prev_a), int(a)), 1.0))
    taus = np.clip(taus, 1e-6, 1e6)
    bias = pheromone_bias * np.log(taus + 1e-12)
    bias = np.clip(bias, -10.0, 10.0)
    return torch.tensor(bias, dtype=torch.float32)


def _compute_operator_loss_from_bank(
    zeros: np.ndarray,
    bank: List[Dict[str, Any]],
    *,
    n_points: int,
    op_sigma: float,
    op_eps: float,
    op_top_k: int,
    seed: int,
) -> Tuple[float, float]:
    if not bank:
        return float("inf"), float("inf")
    bank2 = sorted(bank, key=lambda x: (-float(x["reward"]), float(x["length"])))
    geodesics = [
        {"a_list": list(w["a_list"]), "length": float(w["length"]), "is_hyperbolic": True, "primitive": True}
        for w in bank2[: int(op_top_k)]
    ]
    if not geodesics:
        return float("inf"), float("inf")

    Z = sample_domain(int(n_points), seed=int(seed))
    L, _, _ = build_laplacian(Z, eps=float(op_eps))
    Kmat, used = build_geodesic_kernel(Z, geodesics, sigma=float(op_sigma))
    if used <= 0:
        return float("inf"), float("inf")

    H = -L + Kmat
    H = (H + H.T) * 0.5
    sym_err = float(np.linalg.norm(H - H.T, ord="fro"))

    eigvals = np.linalg.eigh(H, UPLO="U")[0].astype(_DTF, copy=False)
    eigvals.sort()
    if zeros.size < eigvals.size:
        return float("inf"), sym_err
    target = zeros[: eigvals.size].astype(_DTF, copy=False)
    L_spec = float(np.mean((eigvals - target) ** 2))
    return L_spec, sym_err


def _gae_advantages(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float):
    T = rewards.shape[0]
    adv = torch.zeros((T,), dtype=torch.float32, device=rewards.device)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t].item())
        next_value = float(values[t + 1].item()) if t + 1 < values.shape[0] else 0.0
        delta = float(rewards[t].item()) + gamma * next_value * nonterminal - float(values[t].item())
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values[:T]
    return adv, returns


def main() -> None:
    ap = argparse.ArgumentParser(description="V12.4 PPO-style RL policy over Artin symbolic words")
    ap.add_argument("--num_updates", type=int, default=300)
    ap.add_argument("--steps_per_update", type=int, default=1024)
    ap.add_argument("--max_length", type=int, default=8)
    ap.add_argument("--max_power", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--pheromone_bias", type=float, default=0.0)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--eval_operator_every", type=int, default=20)

    ap.add_argument("--lambda_selberg", type=float, default=1.0)
    ap.add_argument("--lambda_spec", type=float, default=1.0)
    ap.add_argument("--lambda_spacing", type=float, default=0.0)
    ap.add_argument("--lambda_selfadj", type=float, default=0.25)

    ap.add_argument("--selberg_sigma", type=float, default=0.5)
    ap.add_argument("--selberg_m_max", type=int, default=6)
    ap.add_argument("--selberg_bank_top_n", type=int, default=250)

    ap.add_argument("--n_points", type=int, default=128)
    ap.add_argument("--op_sigma", type=float, default=0.3)
    ap.add_argument("--op_eps", type=float, default=0.6)
    ap.add_argument("--op_top_k", type=int, default=250)

    ap.add_argument("--target_zeros_path", type=str, default="data/zeta_zeros.txt")
    ap.add_argument("--out_dir", type=str, default="runs/artin_rl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        args.num_updates = 3
        args.steps_per_update = 256
        args.eval_operator_every = 2

    _seed_all(int(args.seed))
    device = get_default_device()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zeros: np.ndarray
    try:
        zeros = _load_zeros(args.target_zeros_path)
        if zeros.size == 0:
            raise ValueError("empty zeros")
    except Exception:
        if not args.dry_run:
            raise
        zeros = (np.arange(1, 5000, dtype=_DTF) * 0.5 + 10.0).astype(_DTF)

    env = ArtinWordEnv(
        max_length=int(args.max_length),
        max_power=int(args.max_power),
        target_zeros_path=str(args.target_zeros_path),
        seed=int(args.seed),
        stop_probability=0.15,
        length_cap=50.0,
    )

    obs_dim = int(env.get_observation().shape[0])
    act_dim = int(env.action_dim)

    net = ArtinPolicyNet(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(args.lr))

    pheromone: Dict[Tuple[int, int], float] = {}
    geodesic_bank: List[Dict[str, Any]] = []
    bank_size = 1000

    best_reward = -1e18
    best_word: List[int] = []
    best_length = 0.0

    history_path = out_dir / "train_history.csv"
    with open(history_path, "w", encoding="utf-8") as f:
        f.write(
            "update,mean_reward,best_reward,mean_length,valid_rate,primitive_rate,entropy,value_loss,policy_loss\n"
        )

    config = vars(args).copy()
    config["device"] = str(device)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    t_start = time.perf_counter()

    pbar = tqdm(range(int(args.num_updates)), desc="ppo-updates", unit="upd")
    for upd in pbar:
        obs_buf = torch.zeros((int(args.steps_per_update), obs_dim), dtype=torch.float32, device=device)
        act_buf = torch.zeros((int(args.steps_per_update),), dtype=torch.int64, device=device)
        logp_buf = torch.zeros((int(args.steps_per_update),), dtype=torch.float32, device=device)
        rew_buf = torch.zeros((int(args.steps_per_update),), dtype=torch.float32, device=device)
        done_buf = torch.zeros((int(args.steps_per_update),), dtype=torch.float32, device=device)
        val_buf = torch.zeros((int(args.steps_per_update),), dtype=torch.float32, device=device)

        ep_rewards: List[float] = []
        ep_lengths: List[float] = []
        ep_valid: List[int] = []
        ep_prim: List[int] = []
        ep_words: List[List[int]] = []

        o = env.reset()
        prev_a = 0

        # rollout
        for t in range(int(args.steps_per_update)):
            obs_t = torch.tensor(o, dtype=torch.float32, device=device)
            with torch.no_grad():
                out = net(obs_t.unsqueeze(0))
                logits = out.logits.squeeze(0)
                value = out.value.squeeze(0)
                bias = _pheromone_logits_bias(pheromone, prev_a, env.actions, float(args.pheromone_bias)).to(device)
                logits = logits + bias
                action, logp, ent = sample_action_and_logp(logits)

            obs_buf[t] = obs_t
            act_buf[t] = action
            logp_buf[t] = logp
            val_buf[t] = value

            o2, r, done, info = env.step(int(action.item()))
            r = float(np.clip(r, -100.0, 100.0))
            if not np.isfinite(r):
                r = -10.0
                done = True

            rew_buf[t] = float(r)
            done_buf[t] = 1.0 if done else 0.0

            prev_a = int(env.prev_a)
            o = o2

            if done:
                inf = info.get("info", {})
                word = list(inf.get("a_list", []))
                ell = float(inf.get("length", 0.0))
                is_hyp = bool(inf.get("is_hyperbolic", False))
                prim = bool(inf.get("primitive", False))

                ep_words.append(word)
                ep_lengths.append(ell)
                ep_valid.append(1 if is_hyp else 0)
                ep_prim.append(1 if prim else 0)

                ep_reward = float(np.sum(rew_buf[max(0, t - env.max_length + 1) : t + 1].detach().cpu().numpy()))
                ep_rewards.append(ep_reward)

                # bank update (episode-level, proxy reward)
                if is_hyp and prim and np.isfinite(ell) and ell > 0.0:
                    geodesic_bank.append({"a_list": word, "length": ell, "reward": ep_reward})
                    geodesic_bank.sort(key=lambda x: (-float(x["reward"]), float(x["length"])))
                    geodesic_bank = geodesic_bank[:bank_size]

                # pheromone update with episode reward
                if float(args.pheromone_bias) > 0.0:
                    _update_pheromone(
                        pheromone,
                        [word] if word else [],
                        reward=float(ep_reward),
                        rho=float(args.rho),
                    )

                o = env.reset()
                prev_a = 0

        # exact end-of-update evaluation (optional)
        L_sel = float("inf")
        L_spec = float("inf")
        sym_err = float("inf")
        if (upd + 1) % max(1, int(args.eval_operator_every)) == 0:
            if geodesic_bank:
                topN = min(int(args.selberg_bank_top_n), len(geodesic_bank))
                lengths = np.array([float(x["length"]) for x in geodesic_bank[:topN]], dtype=_DTF)
                if lengths.size > 0:
                    L_sel = float(
                        compute_selberg_loss(
                            lengths,
                            zeros,
                            sigma=float(args.selberg_sigma),
                            m_max=int(args.selberg_m_max),
                        )
                    )
                L_spec, sym_err = _compute_operator_loss_from_bank(
                    zeros=zeros,
                    bank=geodesic_bank,
                    n_points=int(args.n_points),
                    op_sigma=float(args.op_sigma),
                    op_eps=float(args.op_eps),
                    op_top_k=int(args.op_top_k),
                    seed=int(args.seed),
                )

        # add terminal shaping via exact loss snapshot (bounded)
        exact_term = 0.0
        if np.isfinite(L_sel) or np.isfinite(L_spec) or np.isfinite(sym_err):
            L_total = (
                float(args.lambda_selberg) * (L_sel if np.isfinite(L_sel) else 0.0)
                + float(args.lambda_spec) * (L_spec if np.isfinite(L_spec) else 0.0)
                + float(args.lambda_spacing) * 0.0
                + float(args.lambda_selfadj) * (sym_err if np.isfinite(sym_err) else 0.0)
            )
            exact_term = float(-np.log1p(max(L_total, 0.0)))
            exact_term = float(np.clip(exact_term, -5.0, 5.0))
            rew_buf += exact_term / float(max(1, int(args.steps_per_update)))

        # compute advantages
        with torch.no_grad():
            last_v = net(torch.tensor(env.get_observation(), dtype=torch.float32, device=device).unsqueeze(0)).value
        values_ext = torch.cat([val_buf, last_v.detach().view(1).to(device)], dim=0)
        adv, ret = _gae_advantages(
            rewards=rew_buf,
            values=values_ext,
            dones=done_buf,
            gamma=float(args.gamma),
            lam=float(args.gae_lambda),
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO update
        batch_size = int(args.steps_per_update)
        idx = torch.randperm(batch_size, device=device)
        mini = max(64, batch_size // 8)
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []

        for start in range(0, batch_size, mini):
            mb = idx[start : start + mini]
            obs_mb = obs_buf[mb]
            act_mb = act_buf[mb]
            old_logp_mb = logp_buf[mb]
            adv_mb = adv[mb]
            ret_mb = ret[mb]

            out = net(obs_mb)
            logits = out.logits
            # no pheromone bias in training objective (policy should learn it); bias only for sampling
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act_mb)
            entropy = dist.entropy().mean()
            ratio = torch.exp(logp - old_logp_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - float(args.clip_eps), 1.0 + float(args.clip_eps)) * adv_mb
            policy_loss = -(torch.min(surr1, surr2)).mean()

            value = out.value
            value_loss = 0.5 * ((value - ret_mb) ** 2).mean()

            loss = policy_loss + float(args.value_coef) * value_loss - float(args.entropy_coef) * entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), float(args.max_grad_norm))
            opt.step()

            policy_losses.append(float(policy_loss.detach().cpu().item()))
            value_losses.append(float(value_loss.detach().cpu().item()))
            entropies.append(float(entropy.detach().cpu().item()))

        mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        best_ep_reward = float(np.max(ep_rewards)) if ep_rewards else -1e9
        mean_len = float(np.mean([l for l in ep_lengths if np.isfinite(l)])) if ep_lengths else 0.0
        valid_rate = float(np.mean(ep_valid)) if ep_valid else 0.0
        prim_rate = float(np.mean(ep_prim)) if ep_prim else 0.0

        if best_ep_reward > best_reward:
            best_reward = best_ep_reward
            if ep_rewards:
                bi = int(np.argmax(ep_rewards))
                best_word = ep_words[bi] if bi < len(ep_words) else best_word
                best_length = ep_lengths[bi] if bi < len(ep_lengths) else best_length

        pol_l = float(np.mean(policy_losses)) if policy_losses else 0.0
        val_l = float(np.mean(value_losses)) if value_losses else 0.0
        ent_m = float(np.mean(entropies)) if entropies else 0.0

        elapsed = time.perf_counter() - t_start
        pbar.set_postfix({"meanR": f"{mean_reward:.3g}", "bestR": f"{best_reward:.3g}", "valid": f"{valid_rate:.2f}"})

        print(
            f"[upd {upd+1}/{int(args.num_updates)}] meanR={mean_reward:.6g} bestR={best_reward:.6g} "
            f"valid={valid_rate:.3g} prim={prim_rate:.3g} best_word={best_word} best_ell={best_length:.6g} "
            f"pi_loss={pol_l:.6g} v_loss={val_l:.6g} ent={ent_m:.6g} elapsed={elapsed:.3g}s",
            flush=True,
        )

        with open(history_path, "a", encoding="utf-8") as f:
            f.write(
                f"{upd+1},{mean_reward},{best_reward},{mean_len},{valid_rate},{prim_rate},{ent_m},{val_l},{pol_l}\n"
            )

        # persist periodically
        if (upd + 1) % 10 == 0 or (upd + 1) == int(args.num_updates):
            torch.save({"state_dict": net.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim}, out_dir / "policy.pt")
            with open(out_dir / "best_words.json", "w", encoding="utf-8") as f:
                json.dump({"best_reward": best_reward, "best_word": best_word, "best_length": best_length}, f, indent=2)
            with open(out_dir / "geodesic_bank.json", "w", encoding="utf-8") as f:
                json.dump({"bank": geodesic_bank[:1000]}, f, indent=2)

    # final save
    torch.save({"state_dict": net.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim}, out_dir / "policy.pt")
    with open(out_dir / "best_words.json", "w", encoding="utf-8") as f:
        json.dump({"best_reward": best_reward, "best_word": best_word, "best_length": best_length}, f, indent=2)
    with open(out_dir / "geodesic_bank.json", "w", encoding="utf-8") as f:
        json.dump({"bank": geodesic_bank[:1000]}, f, indent=2)

    if args.dry_run:
        # verify required outputs exist
        required = [
            out_dir / "policy.pt",
            out_dir / "train_history.csv",
            out_dir / "best_words.json",
            out_dir / "geodesic_bank.json",
            out_dir / "config.json",
        ]
        for p in required:
            if not p.exists():
                raise RuntimeError(f"dry_run missing output: {p}")


if __name__ == "__main__":
    main()

