import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from collections import deque
import matplotlib
matplotlib.use('Agg')  # headless backend (no GUI needed)
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """Shared two‑layer MLP with separate policy (actor) and value (critic) heads."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, act_dim)
        self.value_head = nn.Linear(last, 1)

    def forward(self, x):
        feat = self.shared(x)
        return self.policy_head(feat), self.value_head(feat)

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)

    def evaluate_actions(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value.squeeze(-1)


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """Generalised Advantage Estimation."""
    adv = []
    gae = 0.0
    values = np.append(values, next_value)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv.insert(0, gae)
    ret = adv + values[:-1].tolist()
    return np.array(adv, dtype=np.float32), np.array(ret, dtype=np.float32)


def ppo_update(model, optimizer, clip_eps, obs, actions, old_log_probs, returns, advantages,
               batch_size, epochs, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.int64, device=device)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = obs.size(0)
    for _ in range(epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            b_obs = obs[batch_idx]
            b_actions = actions[batch_idx]
            b_old_log = old_log_probs[batch_idx]
            b_returns = returns[batch_idx]
            b_adv = advantages[batch_idx]

            log_probs, entropy, values = model.evaluate_actions(b_obs, b_actions)
            ratio = torch.exp(log_probs - b_old_log)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, b_returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


def train(seed=42,
          env_id="CartPole-v1",
          total_timesteps=500_000,
          rollout_len=2048,
          update_epochs=10,
          batch_size=64,
          gamma=0.99,
          lam=0.95,
          clip_eps=0.2,
          lr=3e-4,
          hidden_sizes=(64, 64),
          plot=True):
    """Train PPO for exactly `total_timesteps` and return the trained model."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim, hidden_sizes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    obs = env.reset()[0]
    ep_rewards = deque(maxlen=100)
    ep_reward = 0.0
    timesteps = 0

    # rollout storage
    ro_obs, ro_actions, ro_logp, ro_rewards, ro_dones, ro_values = ([] for _ in range(6))

    # progress logging
    prog_timesteps, prog_avg_r = [], []

    while timesteps < total_timesteps:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, logp, _, value = model.act(obs_t)
        next_obs, reward, done, truncated, _ = env.step(action.item())
        terminated = done or truncated

        # store
        ro_obs.append(obs)
        ro_actions.append(action.item())
        ro_logp.append(logp.item())
        ro_rewards.append(reward)
        ro_dones.append(terminated)
        ro_values.append(value.item())

        ep_reward += reward
        obs = next_obs
        timesteps += 1

        if terminated:
            obs = env.reset()[0]
            ep_rewards.append(ep_reward)
            ep_reward = 0.0

        # when rollout buffer full -> update
        if len(ro_obs) == rollout_len:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, _, _, next_val = model.act(obs_t)
            adv, ret = compute_gae(ro_rewards, ro_values, ro_dones, next_val.item(), gamma, lam)

            ppo_update(model, optimizer, clip_eps,
                       ro_obs, ro_actions, ro_logp, ret, adv,
                       batch_size, update_epochs)

            # clear buffer
            ro_obs.clear(); ro_actions.clear(); ro_logp.clear();
            ro_rewards.clear(); ro_dones.clear(); ro_values.clear()

            # log moving‑average every update when buffer fills and enough episodes gathered
            if len(ep_rewards) == ep_rewards.maxlen:
                avg_r = np.mean(ep_rewards)
                prog_timesteps.append(timesteps)
                prog_avg_r.append(avg_r)
                print(f"Timesteps: {timesteps}\tAvg Reward (100‑episode MA): {avg_r:.2f}")

    env.close()

    if plot and prog_timesteps:
        plt.figure()
        plt.plot(prog_timesteps, prog_avg_r)
        plt.xlabel('Timesteps')
        plt.ylabel('Average Reward (100‑episode MA)')
        plt.title('PPO on CartPole‑v1')
        plt.grid(True)
        plt.savefig('learning_curve.png')
        print('Learning curve saved as learning_curve.png')

    return model


if __name__ == "__main__":
    train()
