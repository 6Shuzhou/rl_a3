#!/usr/bin/env python3
"""
Proximal Policy Optimization (PPO‑Clip) on CartPole‑v1, PyTorch 2.x
Author: ChatGPT (May 2025)
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from collections import deque

# ───────────────────────────────────────────
# Hyper‑parameters (tuned for CartPole‑v1)
# ───────────────────────────────────────────
ENV_ID           = "CartPole-v1"
TOTAL_UPDATES    = 1_000             # PPO updates
STEPS_PER_ROLLOUT= 2_048             # timesteps of on‑policy data per update
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
CLIP_EPS         = 0.2
ENTROPY_COEF     = 0.01
VALUE_COEF       = 0.5
LR               = 3e-4
EPOCHS           = 4                 # SGD passes per rollout
MINIBATCH_SIZE   = 64
TARGET_REWARD    = 475               # CartPole considered solved at 475
SOLVED_WINDOW    = 20                # consecutive episodes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────────
# Actor‑Critic Network
# ───────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),      nn.Tanh(),
        )
        self.pi = nn.Linear(64, act_dim)
        self.v  = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.pi(x), self.v(x).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist  = Categorical(logits=logits)
        a     = dist.sample()
        logp  = dist.log_prob(a)
        return a.cpu().numpy(), logp, value

    def evaluate(self, obs: torch.Tensor, a: torch.Tensor):
        logits, value = self.forward(obs)
        dist  = Categorical(logits=logits)
        logp  = dist.log_prob(a)
        entropy = dist.entropy()
        return logp, entropy, value

# ───────────────────────────────────────────
# Storage helper for a rollout
# ───────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int):
        self.obs      = np.zeros((size, obs_dim), np.float32)
        self.actions  = np.zeros(size, np.int64)
        self.rewards  = np.zeros(size, np.float32)
        self.dones    = np.zeros(size, np.bool_)
        self.logp     = np.zeros(size, np.float32)
        self.values   = np.zeros(size, np.float32)
        self.ptr = 0; self.max_size = size

    def store(self, obs, act, rew, done, logp, val):
        self.obs[self.ptr]     = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.dones[self.ptr]   = done
        self.logp[self.ptr]    = logp
        self.values[self.ptr]  = val
        self.ptr += 1

    def ready(self): return self.ptr == self.max_size

    def compute_returns_adv(self, last_value: float, gamma, lam):
        """GAE‑λ advantage + discounted returns."""
        adv = np.zeros_like(self.rewards)
        gae = 0
        for t in reversed(range(self.max_size)):
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma*last_value*non_terminal - self.values[t]
            gae   = delta + gamma*lam*non_terminal*gae
            adv[t] = gae
            last_value = self.values[t]
        returns = adv + self.values
        # normalise advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.returns, self.adv = returns, adv

    def get_tensors(self):
        return (
            torch.tensor(self.obs, device=device),
            torch.tensor(self.actions, device=device),
            torch.tensor(self.logp, device=device),
            torch.tensor(self.returns, device=device),
            torch.tensor(self.adv, device=device),
        )

# ───────────────────────────────────────────
# Training loop
# ───────────────────────────────────────────
def main():
    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model   = ActorCritic(obs_dim, act_dim).to(device)
    optimiser = optim.Adam(model.parameters(), lr=LR)

    reward_queue = deque(maxlen=SOLVED_WINDOW)
    global_step  = 0

    for update in range(1, TOTAL_UPDATES + 1):
        buffer = RolloutBuffer(STEPS_PER_ROLLOUT, obs_dim)
        obs, _      = env.reset()
        ep_return   = 0.0                    # <<< NEW
        done        = False

        # ─ Collect on‑policy data ─
        while not buffer.ready():
            obs_t = torch.from_numpy(obs).float().to(device)
            action, logp, value = model.act(obs_t)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())

            done_flag  = terminated or truncated
            buffer.store(obs, action, reward, done_flag, logp.item(), value.item())

            obs        = next_obs
            ep_return += reward              # <<< NEW
            global_step += 1

            if done_flag:
                reward_queue.append(ep_return)   # <<< NEW
                obs, _    = env.reset()
                ep_return = 0.0                  # <<< NEW

        # bootstrap final value for GAE
        obs_t = torch.from_numpy(obs).float().to(device)
        _, last_val = model.forward(obs_t)
        buffer.compute_returns_adv(last_val.item(), GAMMA, GAE_LAMBDA)

        # ─ PPO update ─
        obs_b, act_b, logp_b, ret_b, adv_b = buffer.get_tensors()
        B = STEPS_PER_ROLLOUT
        for _ in range(EPOCHS):
            idx = torch.randperm(B)  # shuffle
            for start in range(0, B, MINIBATCH_SIZE):
                mb = idx[start:start + MINIBATCH_SIZE]
                mb_obs, mb_act     = obs_b[mb], act_b[mb]
                mb_old_logp, mb_ret= logp_b[mb], ret_b[mb]
                mb_adv             = adv_b[mb]

                logp, entropy, value = model.evaluate(mb_obs, mb_act)
                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = (mb_ret - value).pow(2).mean()
                entropy_loss= -entropy.mean()

                loss = policy_loss + VALUE_COEF*value_loss + ENTROPY_COEF*entropy_loss
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        # ─ Logging ─
        if reward_queue:
            avg_r = np.mean(reward_queue)
            print(f"Update {update:4d} │ Steps {global_step:7d} │ "
                  f"AvgR {avg_r:6.1f} │ Advσ {adv_b.std():.3f}")
            if avg_r >= TARGET_REWARD and len(reward_queue) == SOLVED_WINDOW:
                print(f"Environment solved in {update} updates!")
                break

    env.close()

if __name__ == "__main__":
    main()
