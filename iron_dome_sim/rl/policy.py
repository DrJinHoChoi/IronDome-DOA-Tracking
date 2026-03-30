"""PPO Policy Network for RL-based Track Management.

Tiny policy network (~10KB) designed for Edge deployment.
Uses PPO (Proximal Policy Optimization) for stable training.

Author: Jin Ho Choi
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class TrackPolicy(nn.Module):
    """Actor-Critic policy for track management.

    Input: observation (dim=12)
    Output: action mean + std (dim=3), value (dim=1)

    Architecture: MLP with 2 hidden layers.
    Total parameters: ~2K (fits in <10KB INT8).
    """

    def __init__(self, obs_dim=12, act_dim=3, hidden=32):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Actor head (mean + log_std)
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic head
        self.critic = nn.Linear(hidden, 1)

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        """Forward pass.

        Returns:
            action_mean, action_std, value
        """
        features = self.shared(obs)
        mean = torch.tanh(self.actor_mean(features))  # [-1, 1]
        std = torch.exp(self.actor_log_std.clamp(-3, 0.5))
        value = self.critic(features)
        return mean, std, value

    def get_action(self, obs, deterministic=False):
        """Sample action from policy.

        Args:
            obs: np.array (obs_dim,) or torch.Tensor
            deterministic: If True, return mean action.

        Returns:
            action: np.array (act_dim,)
            log_prob: float
            value: float
        """
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).unsqueeze(0)

        mean, std, value = self(obs)

        if deterministic:
            action = mean
            log_prob = torch.zeros(1)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            action = action.clamp(-1, 1)
            log_prob = dist.log_prob(action).sum(-1)

        return (action.squeeze(0).detach().numpy(),
                log_prob.item(),
                value.item())

    def evaluate(self, obs, actions):
        """Evaluate actions for PPO update.

        Args:
            obs: (batch, obs_dim)
            actions: (batch, act_dim)

        Returns:
            log_probs, values, entropy
        """
        mean, std, values = self(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, values.squeeze(-1), entropy

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class RolloutBuffer:
    """Simple rollout buffer for PPO."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def add(self, obs, action, reward, log_prob, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, gamma=0.99, gae_lambda=0.95):
        """Compute GAE advantages and returns."""
        n = len(self.rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        gae = 0
        next_value = 0

        for t in reversed(range(n)):
            if self.dones[t]:
                next_value = 0
                gae = 0

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
            next_value = self.values[t]

        return advantages, returns

    def get_tensors(self, advantages, returns):
        """Convert to PyTorch tensors."""
        obs = torch.FloatTensor(np.array(self.obs))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        adv = torch.FloatTensor(advantages)
        ret = torch.FloatTensor(returns)

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return obs, actions, old_log_probs, adv, ret

    def clear(self):
        self.__init__()


class PPOTrainer:
    """PPO trainer for track management policy.

    Args:
        policy: TrackPolicy instance.
        lr: Learning rate.
        clip_ratio: PPO clipping ratio.
        epochs: PPO update epochs per rollout.
        value_coef: Value loss coefficient.
        entropy_coef: Entropy bonus coefficient.
    """

    def __init__(self, policy, lr=3e-4, clip_ratio=0.2, epochs=4,
                 value_coef=0.5, entropy_coef=0.01):
        self.policy = policy
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    def update(self, buffer):
        """PPO update from rollout buffer.

        Returns:
            dict with loss components.
        """
        advantages, returns = buffer.compute_returns()
        obs, actions, old_log_probs, adv, ret = buffer.get_tensors(
            advantages, returns)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.epochs):
            log_probs, values, entropy = self.policy.evaluate(obs, actions)

            # PPO clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio,
                                1 + self.clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, ret)

            # Entropy bonus
            entropy_bonus = entropy.mean()

            # Total loss
            loss = (policy_loss +
                    self.value_coef * value_loss -
                    self.entropy_coef * entropy_bonus)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_bonus.item()

        n = self.epochs
        return {
            'policy_loss': total_policy_loss / n,
            'value_loss': total_value_loss / n,
            'entropy': total_entropy / n,
        }
