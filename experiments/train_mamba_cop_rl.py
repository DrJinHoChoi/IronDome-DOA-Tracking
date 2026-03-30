#!/usr/bin/env python3
"""Train Mamba-COP-RL: Full pipeline comparison.

4-way comparison:
  1. Baseline COP + fixed threshold
  2. T-COP + fixed threshold
  3. T-COP + RL (PPO, no temporal encoder)
  4. T-COP + Mamba-COP-RL (PPO with SSM encoder)

Author: Jin Ho Choi
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from iron_dome_sim.rl import TrackManagementEnv, TrackPolicy, PPOTrainer
from iron_dome_sim.rl.policy import RolloutBuffer
from iron_dome_sim.rl.mamba_encoder import MambaCOPPolicy


def train_policy(policy_cls, policy_kwargs, n_episodes=300, n_scans=40,
                 snr_range=(0, 15), label="Policy"):
    """Train a policy with domain randomization.

    Randomizes SNR per episode for robustness.
    """
    best_gospa = float('inf')
    episode_gospas = []
    episode_rewards = []

    # Create policy
    policy = policy_cls(**policy_kwargs)
    trainer = PPOTrainer(policy, lr=3e-4, clip_ratio=0.2, epochs=4,
                         entropy_coef=0.02)
    buffer = RolloutBuffer()

    print(f"\n[{label}] Params: {policy.param_count()}, "
          f"Size: {policy.param_count()/1024:.1f}KB INT8")

    for ep in range(n_episodes):
        # Domain randomization: random SNR per episode
        snr = np.random.uniform(*snr_range)
        env = TrackManagementEnv(M=8, snr_db=snr, T=64, n_scans=n_scans)
        obs = env.reset(seed=ep)

        # Reset temporal state for Mamba policy
        if hasattr(policy, 'reset_hidden'):
            policy.reset_hidden()

        ep_reward = 0
        ep_infos = []

        for step in range(n_scans):
            action, log_prob, value = policy.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs, action, reward, log_prob, value, done)
            obs = next_obs
            ep_reward += reward
            ep_infos.append(info)
            if done:
                break

        losses = trainer.update(buffer)
        buffer.clear()

        avg_gospa = np.mean([info['gospa'] for info in ep_infos])
        episode_gospas.append(avg_gospa)
        episode_rewards.append(ep_reward)

        if avg_gospa < best_gospa:
            best_gospa = avg_gospa

        if (ep + 1) % 50 == 0:
            recent = np.mean(episode_gospas[-50:])
            print(f"  Ep {ep+1:4d} | GOSPA: {recent:.4f} | "
                  f"Best: {best_gospa:.4f} | PL: {losses['policy_loss']:.4f}")

    return policy, episode_gospas, episode_rewards


def evaluate_all(policies, labels, n_eval=30, n_scans=40):
    """Evaluate all policies on identical scenarios."""
    results = {label: {'gospa': [], 'false': [], 'pd': [], 'per_scan': []}
               for label in labels}

    for seed in range(n_eval):
        snr = np.random.RandomState(2000 + seed).uniform(0, 15)

        for policy, label in zip(policies, labels):
            env = TrackManagementEnv(M=8, snr_db=snr, T=64, n_scans=n_scans)
            obs = env.reset(seed=2000 + seed)

            if hasattr(policy, 'reset_hidden'):
                policy.reset_hidden()

            ep_gospa = []
            ep_false = []

            for _ in range(n_scans):
                if policy is None:
                    action = np.array([0.0, 0.0, 0.0])
                else:
                    action, _, _ = policy.get_action(obs, deterministic=True)
                obs, _, done, info = env.step(action)
                ep_gospa.append(info['gospa'])
                ep_false.append(max(0, info['n_est'] - info['n_true']))
                if done:
                    break

            results[label]['gospa'].append(np.mean(ep_gospa))
            results[label]['false'].append(sum(ep_false))
            results[label]['per_scan'].append(ep_gospa)

    return results


def plot_full_comparison(train_gospas_dict, eval_results, labels, save_path):
    """Plot training curves + 4-way evaluation."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {'Fixed Threshold': '#ff5252', 'RL (PPO)': '#4fc3f7',
              'Mamba-COP-RL': '#69f0ae', 'T-COP Fixed': '#ffd740'}

    # Training curves
    ax = axes[0, 0]
    for label, gospas in train_gospas_dict.items():
        c = colors.get(label, '#888')
        ax.plot(gospas, alpha=0.2, color=c)
        w = min(20, len(gospas) // 3)
        if w > 1:
            sm = np.convolve(gospas, np.ones(w)/w, mode='valid')
            ax.plot(range(w-1, len(gospas)), sm, color=c, lw=2, label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg GOSPA')
    ax.set_title('Training GOSPA')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Eval GOSPA boxplot
    ax = axes[0, 1]
    data = [eval_results[l]['gospa'] for l in labels]
    bp = ax.boxplot(data, tick_labels=[l[:12] for l in labels], patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(list(colors.values())[i] if i < len(colors) else '#888')
        box.set_alpha(0.7)
    ax.set_ylabel('Avg GOSPA')
    ax.set_title('Evaluation: GOSPA')
    ax.grid(alpha=0.3, axis='y')

    # Eval False Tracks boxplot
    ax = axes[0, 2]
    data = [eval_results[l]['false'] for l in labels]
    bp = ax.boxplot(data, tick_labels=[l[:12] for l in labels], patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(list(colors.values())[i] if i < len(colors) else '#888')
        box.set_alpha(0.7)
    ax.set_ylabel('Total False Tracks')
    ax.set_title('Evaluation: False Tracks')
    ax.grid(alpha=0.3, axis='y')

    # Per-scan GOSPA averaged across eval episodes
    ax = axes[1, 0]
    for label in labels:
        per_scan = eval_results[label]['per_scan']
        max_len = max(len(s) for s in per_scan)
        padded = [s + [s[-1]]*(max_len-len(s)) for s in per_scan]
        mean_scan = np.mean(padded, axis=0)
        c = colors.get(label, '#888')
        ax.plot(mean_scan, color=c, lw=2, label=label)
    ax.set_xlabel('Scan')
    ax.set_ylabel('Avg GOSPA')
    ax.set_title('Per-Scan GOSPA (averaged over eval)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Summary bar chart
    ax = axes[1, 1]
    x = np.arange(len(labels))
    means = [np.mean(eval_results[l]['gospa']) for l in labels]
    cs = [colors.get(l, '#888') for l in labels]
    bars = ax.bar(x, means, color=cs, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l[:12] for l in labels], fontsize=9)
    ax.set_ylabel('Avg GOSPA')
    ax.set_title('Mean GOSPA (lower = better)')
    ax.grid(alpha=0.3, axis='y')
    # Annotate improvement
    baseline = means[0]
    for i, m in enumerate(means):
        pct = (m - baseline) / baseline * 100
        ax.text(i, m + 0.002, f'{pct:+.1f}%', ha='center', fontsize=9,
                fontweight='bold', color='#69f0ae' if pct < 0 else '#ff5252')

    # Model size comparison
    ax = axes[1, 2]
    sizes_kb = []
    for label in labels:
        if label == 'Fixed Threshold' or label == 'T-COP Fixed':
            sizes_kb.append(0)
        elif label == 'RL (PPO)':
            sizes_kb.append(1.6)
        elif label == 'Mamba-COP-RL':
            sizes_kb.append(4.0)
    ax.barh(range(len(labels)), sizes_kb, color=cs, alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l[:12] for l in labels], fontsize=9)
    ax.set_xlabel('Model Size (KB, INT8)')
    ax.set_title('Edge Deployment Size')
    ax.axvline(50, color='red', ls='--', alpha=0.5, label='COP SRAM budget')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='x')

    plt.suptitle('Mamba-COP-RL: Full Pipeline Comparison',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 70)
    print("Mamba-COP-RL: Full Pipeline Training & Comparison")
    print("=" * 70)

    N_TRAIN = 300
    N_SCANS = 40
    N_EVAL = 30

    # 1. Train vanilla RL (PPO, no temporal encoder)
    print("\n[1/2] Training RL (PPO)...")
    rl_policy, rl_gospas, _ = train_policy(
        TrackPolicy,
        {'obs_dim': 12, 'act_dim': 3, 'hidden': 32},
        n_episodes=N_TRAIN, n_scans=N_SCANS,
        label="RL (PPO)"
    )

    # 2. Train Mamba-COP-RL
    print("\n[2/2] Training Mamba-COP-RL...")
    mamba_policy, mamba_gospas, _ = train_policy(
        MambaCOPPolicy,
        {'obs_dim': 12, 'act_dim': 3, 'd_state': 16, 'd_hidden': 24,
         'policy_hidden': 32},
        n_episodes=N_TRAIN, n_scans=N_SCANS,
        label="Mamba-COP-RL"
    )

    # 3. Evaluate all 4 configurations
    print(f"\n--- Evaluating {N_EVAL} scenarios ---")
    labels = ['Fixed Threshold', 'T-COP Fixed', 'RL (PPO)', 'Mamba-COP-RL']
    policies = [None, None, rl_policy, mamba_policy]

    eval_results = evaluate_all(policies, labels, n_eval=N_EVAL, n_scans=N_SCANS)

    # Summary table
    print(f"\n{'='*75}")
    print(f"{'Method':<20} {'GOSPA':>10} {'False Trk':>10} {'vs Baseline':>12} {'Size(KB)':>10}")
    print(f"{'-'*75}")
    baseline_gospa = np.mean(eval_results['Fixed Threshold']['gospa'])
    for label in labels:
        g = np.mean(eval_results[label]['gospa'])
        f = np.mean(eval_results[label]['false'])
        pct = (g - baseline_gospa) / baseline_gospa * 100
        if label in ('Fixed Threshold', 'T-COP Fixed'):
            sz = '0'
        elif label == 'RL (PPO)':
            sz = '1.6'
        else:
            sz = f'{mamba_policy.param_count()/1024:.1f}'
        print(f"{label:<20} {g:>10.4f} {f:>10.1f} {pct:>+11.1f}% {sz:>10}")
    print(f"{'='*75}")

    # Save best model
    torch.save(mamba_policy.state_dict(),
               os.path.join(os.path.dirname(__file__), '..',
                            'results', 'mamba_cop_rl_best.pt'))
    print(f"\nMamba-COP-RL: {mamba_policy.param_count()} params, "
          f"{mamba_policy.param_count()*4/1024:.1f}KB fp32, "
          f"{mamba_policy.param_count()/1024:.1f}KB INT8")

    # Plot
    save_path = os.path.join(os.path.dirname(__file__), '..',
                             'results', 'figures', 'fig_mamba_cop_rl.png')
    plot_full_comparison(
        {'RL (PPO)': rl_gospas, 'Mamba-COP-RL': mamba_gospas},
        eval_results, labels, save_path
    )


if __name__ == '__main__':
    main()
