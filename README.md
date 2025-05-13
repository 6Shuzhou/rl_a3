# Reinforcement Learning Experiments: CartPole-v1

A minimal PyTorch implementation of Proximal Policy Optimization (PPO) applied to OpenAI Gym environments, with example usage on CartPole-v1.

## Repository Structure

- `ppo.py`  
  Core implementation of PPO, including:
  - **ActorCritic**: Shared feature trunk with separate policy and value heads.
  - **gae**: Generalized Advantage Estimation for computing advantages and returns.
  - **ppo_update**: Clipped surrogate objective, value loss, and entropy bonus.
  - **train**: Training loop for a single experiment, with logging of average rewards.
  - **run_multiple**: Runs multiple seeds, aggregates learning curves, and saves `learning_curve_avg.png`.

- `learning_curve_avg.png`  
  Plot of average reward vs. training steps across multiple runs.

- `requirements.txt`  
  Python dependencies: `gym`, `numpy`, `torch`, `matplotlib`.

## 🚀 Installation

1. **Getting Started**
   ```bash
   git clone https://github.com/6Shuzhou/rl_a3.git
   cd rl_a3
   pip install -r requirements.txt

2. **PPO algorithm**  
   ```bash
   python ppo.py