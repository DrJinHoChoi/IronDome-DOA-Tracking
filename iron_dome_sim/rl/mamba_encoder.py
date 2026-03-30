"""Mamba-style SSM Encoder for Temporal Cumulant Features.

Processes sequence of cumulant matrices across scans using a
selective state space model (S4/Mamba architecture).

The hidden state accumulates temporal context automatically:
  h_t = f(h_{t-1}, cumulant_t)

This replaces the simple EMA in TemporalCOP with a learned
temporal encoder that can selectively attend to important
events (births, deaths, crossings).

Designed for Edge deployment: ~5-15KB INT8.

Author: Jin Ho Choi
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMBlock(nn.Module):
    """Selective State Space Model block (Mamba-inspired).

    Implements the core S6 selective scan:
        h_t = A_t * h_{t-1} + B_t * x_t
        y_t = C_t * h_t

    Where A_t, B_t, C_t are input-dependent (selective).
    This allows the model to decide what to remember/forget
    based on the current input.

    Args:
        d_input: Input dimension.
        d_state: SSM hidden state dimension.
        d_output: Output dimension.
    """

    def __init__(self, d_input, d_state=16, d_output=None):
        super().__init__()
        d_output = d_output or d_input

        self.d_state = d_state

        # Input projection
        self.proj_in = nn.Linear(d_input, d_input)

        # Selective parameters (input-dependent)
        self.proj_A = nn.Linear(d_input, d_state)  # → log(A) diagonal
        self.proj_B = nn.Linear(d_input, d_state)   # → B
        self.proj_C = nn.Linear(d_input, d_state)   # → C
        self.proj_dt = nn.Linear(d_input, 1)         # → discretization step

        # Output projection
        self.proj_out = nn.Linear(d_input, d_output)

        # Layer norm
        self.norm = nn.LayerNorm(d_input)

    def forward(self, x_seq, h_prev=None):
        """Process sequence through selective SSM.

        Args:
            x_seq: (batch, seq_len, d_input) or (seq_len, d_input)
            h_prev: Previous hidden state (batch, d_state) or None.

        Returns:
            y_seq: Output sequence, same shape as x_seq.
            h_final: Final hidden state.
        """
        squeeze = False
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(0)
            squeeze = True

        B, L, D = x_seq.shape

        if h_prev is None:
            h = torch.zeros(B, self.d_state, device=x_seq.device)
        else:
            h = h_prev

        x = self.norm(x_seq)
        x = F.silu(self.proj_in(x))  # (B, L, D)

        outputs = []
        for t in range(L):
            x_t = x[:, t, :]  # (B, D)

            # Selective parameters
            log_A = -F.softplus(self.proj_A(x_t))  # negative → stable
            A_t = torch.exp(log_A)  # (B, d_state), values in (0, 1)
            B_t = self.proj_B(x_t)   # (B, d_state)
            C_t = self.proj_C(x_t)   # (B, d_state)
            dt = F.softplus(self.proj_dt(x_t))  # (B, 1)

            # Discretized state transition
            A_disc = torch.exp(log_A * dt)  # (B, d_state)
            B_disc = B_t * dt  # (B, d_state)

            # State update: h_t = A * h_{t-1} + B * mean(x_t)
            h = A_disc * h + B_disc * x_t.mean(dim=-1, keepdim=True)

            # Output: y_t = C * h_t
            y_t = (C_t * h).sum(dim=-1, keepdim=True)  # (B, 1)
            outputs.append(y_t)

        y_seq = torch.stack(outputs, dim=1)  # (B, L, 1)
        # Expand and project
        y_seq = y_seq.expand(-1, -1, D)
        y_seq = self.proj_out(x * y_seq)  # gated output

        # Residual
        y_seq = y_seq + x_seq

        if squeeze:
            y_seq = y_seq.squeeze(0)

        return y_seq, h

    def step(self, x_t, h_prev):
        """Single-step forward (for real-time inference).

        Args:
            x_t: (batch, d_input) or (d_input,)
            h_prev: (batch, d_state) or (d_state,)

        Returns:
            y_t: (batch, d_output)
            h_new: (batch, d_state)
        """
        squeeze = False
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
            h_prev = h_prev.unsqueeze(0) if h_prev.dim() == 1 else h_prev
            squeeze = True

        x = self.norm(x_t.unsqueeze(1)).squeeze(1)
        x = F.silu(self.proj_in(x))

        log_A = -F.softplus(self.proj_A(x))
        A_t = torch.exp(log_A)
        B_t = self.proj_B(x)
        C_t = self.proj_C(x)
        dt = F.softplus(self.proj_dt(x))

        A_disc = torch.exp(log_A * dt)
        B_disc = B_t * dt

        h_new = A_disc * h_prev + B_disc * x.mean(dim=-1, keepdim=True)
        y_t = (C_t * h_new).sum(dim=-1, keepdim=True)

        D_in = x_t.shape[-1]
        y_t = y_t.expand(-1, D_in)
        y_t = self.proj_out(x * y_t) + x_t

        if squeeze:
            y_t = y_t.squeeze(0)
            h_new = h_new.squeeze(0)

        return y_t, h_new


class MambaCOPEncoder(nn.Module):
    """Mamba encoder for COP-DOA temporal feature extraction.

    Processes per-scan features through selective SSM layers
    to produce temporally-enriched observations for the RL policy.

    Architecture:
        Input (obs_dim) → Linear → SSM Block → Linear → Output (obs_dim)

    The SSM hidden state (d_state) serves as temporal memory,
    analogous to NC-SSM's hidden state in keyword spotting.

    Args:
        obs_dim: Observation dimension from TrackManagementEnv.
        d_state: SSM hidden state dimension (temporal memory).
        d_hidden: Internal feature dimension.
    """

    def __init__(self, obs_dim=12, d_state=16, d_hidden=24):
        super().__init__()

        self.d_state = d_state
        self.obs_dim = obs_dim

        # Feature extraction
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, d_hidden),
            nn.SiLU(),
        )

        # Selective SSM block
        self.ssm = SelectiveSSMBlock(d_hidden, d_state, d_hidden)

        # Output projection back to obs_dim
        self.output_proj = nn.Linear(d_hidden, obs_dim)

    def forward(self, obs_seq, h_prev=None):
        """Process observation sequence.

        Args:
            obs_seq: (batch, seq_len, obs_dim) or (seq_len, obs_dim)
            h_prev: Previous SSM hidden state.

        Returns:
            enriched_obs: Same shape as obs_seq, temporally enriched.
            h_final: Final hidden state for next call.
        """
        squeeze = False
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(0)
            squeeze = True

        x = self.input_proj(obs_seq)
        x, h = self.ssm(x, h_prev)
        out = self.output_proj(x)

        # Residual connection to original obs
        out = out + obs_seq

        if squeeze:
            out = out.squeeze(0)

        return out, h

    def step(self, obs_t, h_prev):
        """Single-step (real-time Edge inference).

        Args:
            obs_t: (obs_dim,) current observation.
            h_prev: (d_state,) previous hidden state.

        Returns:
            enriched_obs: (obs_dim,) temporally enriched.
            h_new: (d_state,) updated hidden state.
        """
        if isinstance(obs_t, np.ndarray):
            obs_t = torch.FloatTensor(obs_t)
        if isinstance(h_prev, np.ndarray):
            h_prev = torch.FloatTensor(h_prev)

        x = self.input_proj(obs_t.unsqueeze(0).unsqueeze(0)).squeeze(0)
        y, h = self.ssm.step(x.squeeze(0), h_prev)
        out = self.output_proj(y) + obs_t

        return out, h

    def init_hidden(self):
        """Initialize hidden state (zeros)."""
        return torch.zeros(self.d_state)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class MambaCOPPolicy(nn.Module):
    """Full Mamba-COP-RL policy: SSM encoder + actor-critic.

    Pipeline per scan:
        obs_t → [Mamba Encoder] → enriched_obs → [Actor-Critic] → action, value
                     ↑ h_{t-1}        ↓ h_t (carried to next scan)

    Total: ~4K params, ~4KB INT8 — fits on Cortex-M7.
    """

    def __init__(self, obs_dim=12, act_dim=3, d_state=16, d_hidden=24,
                 policy_hidden=32):
        super().__init__()

        self.obs_dim = obs_dim
        self.d_state = d_state

        # Mamba encoder
        self.encoder = MambaCOPEncoder(obs_dim, d_state, d_hidden)

        # Actor-Critic (same as TrackPolicy but takes enriched obs)
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, policy_hidden),
            nn.Tanh(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(policy_hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(policy_hidden, 1)

        # Hidden state
        self._h = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def reset_hidden(self):
        """Reset hidden state for new episode."""
        self._h = self.encoder.init_hidden()

    def forward(self, obs):
        """Forward for batch evaluation (no temporal state)."""
        features = self.shared(obs)
        mean = torch.tanh(self.actor_mean(features))
        std = torch.exp(self.actor_log_std.clamp(-3, 0.5))
        value = self.critic(features)
        return mean, std, value

    def get_action(self, obs, deterministic=False):
        """Get action with temporal encoding.

        Uses single-step SSM inference to maintain hidden state.
        """
        if isinstance(obs, np.ndarray):
            obs_t = torch.FloatTensor(obs)
        else:
            obs_t = obs

        if self._h is None:
            self._h = self.encoder.init_hidden()

        # Enrich observation with temporal context
        enriched, self._h = self.encoder.step(obs_t, self._h)
        self._h = self._h.detach()

        enriched = enriched.unsqueeze(0)
        mean, std, value = self.forward(enriched)

        if deterministic:
            action = mean
            log_prob = torch.zeros(1)
        else:
            from torch.distributions import Normal
            dist = Normal(mean, std)
            action = dist.sample().clamp(-1, 1)
            log_prob = dist.log_prob(action).sum(-1)

        return (action.squeeze(0).detach().numpy(),
                log_prob.item(),
                value.item())

    def evaluate(self, obs, actions):
        """Evaluate for PPO update (batch, no temporal)."""
        from torch.distributions import Normal
        mean, std, values = self.forward(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, values.squeeze(-1), entropy

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
