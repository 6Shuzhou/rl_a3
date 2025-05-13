import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from collections import deque
import matplotlib
matplotlib.use('Agg')  # non‑interactive backend
import matplotlib.pyplot as plt

# Choose GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """Two‑layer MLP shared trunk with separate policy and value heads."""

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
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), v.squeeze(-1)

    def evaluate(self, obs, actions):
        logits, v = self.forward(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions)
        ent = dist.entropy()
        return logp, ent, v.squeeze(-1)


def gae(rew, val, dones, next_v, gamma=0.99, lam=0.95):
    """Generalised Advantage Estimation (GAE‑λ)."""
    adv, g = [], 0.0
    val = np.append(val, next_v)
    for t in reversed(range(len(rew))):
        delta = rew[t] + gamma * val[t + 1] * (1 - dones[t]) - val[t]
        g = delta + gamma * lam * (1 - dones[t]) * g
        adv.insert(0, g)
    ret = adv + val[:-1].tolist()
    return np.array(adv, np.float32), np.array(ret, np.float32)


def ppo_update(model, opt, clip_eps, obs, act, logp_old, ret, adv,
               batch, epochs, vf_coef=0.5, ent_coef=0.01, max_grad=0.5):
    """One full PPO optimisation phase (multiple minibatch epochs)."""
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    act = torch.tensor(act, dtype=torch.int64, device=device)
    logp_old = torch.tensor(logp_old, dtype=torch.float32, device=device)
    ret = torch.tensor(ret, dtype=torch.float32, device=device)
    adv = torch.tensor(adv, dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = obs.size(0)
    for _ in range(epochs):
        idx = torch.randperm(n)
        for st in range(0, n, batch):
            b = idx[st:st + batch]
            b_obs, b_act = obs[b], act[b]
            b_old, b_ret, b_adv = logp_old[b], ret[b], adv[b]

            logp, ent, v = model.evaluate(b_obs, b_act)
            ratio = torch.exp(logp - b_old)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss = nn.functional.mse_loss(v, b_ret)
            ent_loss = -ent.mean()

            loss = pi_loss + vf_coef * v_loss + ent_coef * ent_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            opt.step()


def train(seed: int = 42,
          env_id: str = "CartPole-v1",
          total_steps: int = 1_000_000,
          rollout: int = 2048,
          epochs: int = 10,
          batch: int = 64,
          gamma: float = 0.99,
          lam: float = 0.95,
          clip_eps: float = 0.1,
          actor_lr: float = 3e-4,
          critic_lr: float = 1e-4,
          hidden=(64, 64),
          run_id: int = 1,
          plot: bool = False):
    """Train PPO for a single run and return (steps, rewards).

    Each element of the returned lists corresponds to *one PPO update*.
    The x‑axis is cumulative environment steps (not update index).
    """
    env = gym.make(env_id)
    env.reset(seed=seed)
    np.random.seed(seed); torch.manual_seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim, hidden).to(device)

    # Separate LR for actor/critic
    actor_params = list(model.shared.parameters()) + list(model.policy.parameters())
    critic_params = model.value.parameters()
    opt = optim.Adam([
        {'params': actor_params, 'lr': actor_lr},
        {'params': critic_params, 'lr': critic_lr}
    ], eps=1e-5)

    obs, _ = env.reset()
    ep_buf = deque(maxlen=100)
    ep_ret, steps = 0.0, 0

    # rollout buffers
    o_buf, a_buf, logp_buf, r_buf, d_buf, v_buf = ([] for _ in range(6))

    t_log, r_log = [], []  # cumulative steps, moving‑average reward
    update_idx = 0

    while steps < total_steps:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a, logp, _, v = model.act(obs_t)
        new_obs, rew, term, trunc, _ = env.step(a.item())
        done = term or trunc

        # store transition
        o_buf.append(obs)
        a_buf.append(a.item())
        logp_buf.append(logp.item())
        r_buf.append(rew)
        d_buf.append(done)
        v_buf.append(v.item())

        ep_ret += rew
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

            o_buf.clear(); a_buf.clear(); logp_buf.clear();
            r_buf.clear(); d_buf.clear(); v_buf.clear()

            if len(ep_buf) == ep_buf.maxlen:
                avg_r = np.mean(ep_buf)
                t_log.append(steps)
                r_log.append(avg_r)
                update_idx += 1
                print(f"Run {run_id} | Upd {update_idx:4d} | Steps {steps:7d} | AvgR(100ep) {avg_r:6.2f}")

    env.close()

    # Optional single‑run plot
    if plot and r_log:
        plt.figure()
        plt.plot(t_log, r_log)
        plt.xlabel('Environment steps')
        plt.ylabel('Average Reward (100‑episode MA)')
        plt.title(f'PPO on {env_id} – Run {run_id}')
        plt.grid(True)
        fname = f'learning_curve_run{run_id}.png'
        plt.savefig(fname)
        print(f'Saved {fname}')

    return t_log, r_log


def run_multiple(repeats: int = 5, **train_kwargs):
    """Run PPO *repeats* times and plot the mean learning curve vs *steps*."""
    steps_list, curves = [], []
    for i in range(repeats):
        print(f"========== Run {i + 1}/{repeats} ==========")
        t_log, r_log = train(seed=train_kwargs.get('seed', 42) + i,
                             run_id=i + 1,
                             plot=False,
                             **{k: v for k, v in train_kwargs.items() if k not in ('seed', 'run_id')})
        steps_list.append(t_log)
        curves.append(r_log)

    # Align lengths (they should already match)
    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]
    steps_axis = steps_list[0][:min_len]
    avg_curve = np.mean(curves, axis=0)

    # Plot mean curve
    plt.figure()
    plt.plot(steps_axis, avg_curve)
    plt.xlabel('Environment steps')
    plt.ylabel('Average Reward (100‑episode MA)')
    plt.title(f'PPO on CartPole‑v1 ')
    plt.grid(True)
    plt.savefig('learning_curve_avg.png')
    print('Saved learning_curve_avg.png')


if __name__ == '__main__':
    # Run 5 times (default) and plot averaged learning curve
    run_multiple(repeats=1, total_steps=1_000_000)