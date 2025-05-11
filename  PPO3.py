import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from collections import deque
import matplotlib
matplotlib.use('Agg')  # headless backend for file output
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """Two‑layer MLP with policy and value heads."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.shared = nn.Sequential(*layers)
        self.policy = nn.Linear(last, act_dim)
        self.value = nn.Linear(last, 1)

    def forward(self, x):
        feat = self.shared(x)
        return self.policy(feat), self.value(feat)

    def act(self, obs):
        logits, v = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), v.squeeze(-1)

    def evaluate(self, obs, actions):
        logits, v = self.forward(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, entropy, v.squeeze(-1)


def gae(rew, val, dones, next_v, gamma=0.99, lam=0.95):
    adv = []
    g = 0.0
    val = np.append(val, next_v)
    for t in reversed(range(len(rew))):
        delta = rew[t] + gamma * val[t + 1] * (1 - dones[t]) - val[t]
        g = delta + gamma * lam * (1 - dones[t]) * g
        adv.insert(0, g)
    ret = adv + val[:-1].tolist()
    return np.array(adv, np.float32), np.array(ret, np.float32)


def ppo_update(model, opt, clip_eps, obs, act, logp_old, ret, adv,
               batch, epochs, vf_coef=0.5, ent_coef=0.01, max_grad=0.5):
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    act = torch.tensor(act, dtype=torch.int64, device=device)
    logp_old = torch.tensor(logp_old, dtype=torch.float32, device=device)
    ret = torch.tensor(ret, dtype=torch.float32, device=device)
    adv = torch.tensor(adv, dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = obs.size(0)
    for _ in range(epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch):
            b = idx[start:start + batch]
            b_obs, b_act = obs[b], act[b]
            b_old, b_ret, b_adv = logp_old[b], ret[b], adv[b]

            logp, ent, v = model.evaluate(b_obs, b_act)
            ratio = torch.exp(logp - b_old)
            s1 = ratio * b_adv
            s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
            pi_loss = -torch.min(s1, s2).mean()
            v_loss = nn.functional.mse_loss(v, b_ret)
            ent_loss = -ent.mean()

            loss = pi_loss + vf_coef * v_loss + ent_coef * ent_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            opt.step()


def train(seed=42,
          env_id="CartPole-v1",
          total_steps=1_000_000,
          rollout=2048,
          epochs=10,
          batch=64,
          gamma=0.99,
          lam=0.95,
          clip_eps=0.2,
          lr=3e-4,
          hidden=(64, 64),
          plot=True):
    env = gym.make(env_id)
    env.reset(seed=seed)
    np.random.seed(seed); torch.manual_seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim, hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    obs, _ = env.reset()
    ep_buf = deque(maxlen=100)
    ep_ret = 0.0
    steps = 0

    # rollout buffers
    o_buf, a_buf, logp_buf, r_buf, d_buf, v_buf = ([] for _ in range(6))

    # progress logs
    t_log, r_log = [], []

    while steps < total_steps:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            act, logp, _, v = model.act(obs_t)
        new_obs, reward, terminated, truncated, _ = env.step(act.item())
        done = terminated or truncated

        # store transition
        o_buf.append(obs)
        a_buf.append(act.item())
        logp_buf.append(logp.item())
        r_buf.append(reward)
        d_buf.append(done)
        v_buf.append(v.item())

        ep_ret += reward
        obs = new_obs
        steps += 1

        if done:
            obs, _ = env.reset()
            ep_buf.append(ep_ret)
            ep_ret = 0.0

        if len(o_buf) == rollout:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, _, _, next_v = model.act(obs_t)
            adv, ret = gae(r_buf, v_buf, d_buf, next_v.item(), gamma, lam)

            ppo_update(model, opt, clip_eps,
                       o_buf, a_buf, logp_buf, ret, adv,
                       batch, epochs)

            # clear rollout buffers
            o_buf.clear(); a_buf.clear(); logp_buf.clear();
            r_buf.clear(); d_buf.clear(); v_buf.clear()

            if len(ep_buf) == ep_buf.maxlen:
                avg_r = np.mean(ep_buf)
                t_log.append(steps)
                r_log.append(avg_r)
                print(f"Steps: {steps}\tAvg Reward (100‑ep MA): {avg_r:.2f}")

    env.close()

    if plot and t_log:
        plt.figure()
        plt.plot(t_log, r_log)
        plt.xlabel('Timesteps')
        plt.ylabel('Average Reward (100‑episode MA)')
        plt.title('PPO on CartPole‑v1 (Gymnasium)')
        plt.grid(True)
        plt.savefig('learning_curve.png')
        print('Saved learning_curve.png')

    return model


if __name__ == '__main__':
    train()
