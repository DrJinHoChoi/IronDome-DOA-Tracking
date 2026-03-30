#!/usr/bin/env python3
"""Train RL-based Track Manager and compare vs fixed-threshold baseline.

Trains a tiny PPO policy (~2K params) that adaptively sets
birth_weight, prune_threshold, and detection_prob per scan.

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


def train(n_episodes=100, n_scans=40, print_every=10):
    """Train PPO policy on random birth-death scenarios."""

    env = TrackManagementEnv(M=8, snr_db=5, T=64, n_scans=n_scans)
    policy = TrackPolicy(obs_dim=env.OBS_DIM, act_dim=env.ACT_DIM, hidden=32)
    trainer = PPOTrainer(policy, lr=3e-4, clip_ratio=0.2, epochs=4)
    buffer = RolloutBuffer()

    print(f"Policy parameters: {policy.param_count()}")
    print(f"Model size (float32): {policy.param_count() * 4 / 1024:.1f} KB")
    print(f"Model size (INT8):    {policy.param_count() / 1024:.1f} KB")
    print()

    episode_rewards = []
    episode_gospas = []
    best_gospa = float('inf')

    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
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

        # PPO update every episode
        losses = trainer.update(buffer)
        buffer.clear()

        avg_gospa = np.mean([info['gospa'] for info in ep_infos])
        avg_n_est = np.mean([info['n_est'] for info in ep_infos])
        avg_n_true = np.mean([info['n_true'] for info in ep_infos])
        episode_rewards.append(ep_reward)
        episode_gospas.append(avg_gospa)

        if avg_gospa < best_gospa:
            best_gospa = avg_gospa
            torch.save(policy.state_dict(), os.path.join(
                os.path.dirname(__file__), '..', 'results', 'rl_policy_best.pt'))

        if (ep + 1) % print_every == 0:
            recent_reward = np.mean(episode_rewards[-print_every:])
            recent_gospa = np.mean(episode_gospas[-print_every:])
            print(f"Ep {ep+1:4d} | Reward: {recent_reward:7.3f} | "
                  f"GOSPA: {recent_gospa:.4f} | Best: {best_gospa:.4f} | "
                  f"Est/True: {avg_n_est:.1f}/{avg_n_true:.1f} | "
                  f"PL: {losses['policy_loss']:.4f} VL: {losses['value_loss']:.4f}")

    return policy, episode_rewards, episode_gospas


def evaluate_comparison(policy, n_eval=20, n_scans=40):
    """Compare RL policy vs fixed-threshold baseline on same scenarios."""

    rl_gospas = []
    rl_false = []
    baseline_gospas = []
    baseline_false = []

    for seed in range(n_eval):
        # RL agent
        env_rl = TrackManagementEnv(M=8, snr_db=5, T=64, n_scans=n_scans)
        obs = env_rl.reset(seed=1000 + seed)
        rl_ep_gospa = []
        rl_ep_false = []
        for _ in range(n_scans):
            action, _, _ = policy.get_action(obs, deterministic=True)
            obs, _, done, info = env_rl.step(action)
            rl_ep_gospa.append(info['gospa'])
            rl_ep_false.append(max(0, info['n_est'] - info['n_true']))
            if done:
                break
        rl_gospas.append(np.mean(rl_ep_gospa))
        rl_false.append(sum(rl_ep_false))

        # Fixed-threshold baseline (action = [0, 0, 0] → mid-range params)
        env_bl = TrackManagementEnv(M=8, snr_db=5, T=64, n_scans=n_scans)
        obs = env_bl.reset(seed=1000 + seed)
        bl_ep_gospa = []
        bl_ep_false = []
        for _ in range(n_scans):
            action = np.array([0.0, 0.0, 0.0])  # fixed mid-range
            obs, _, done, info = env_bl.step(action)
            bl_ep_gospa.append(info['gospa'])
            bl_ep_false.append(max(0, info['n_est'] - info['n_true']))
            if done:
                break
        baseline_gospas.append(np.mean(bl_ep_gospa))
        baseline_false.append(sum(bl_ep_false))

    return {
        'rl_gospa': rl_gospas,
        'rl_false': rl_false,
        'baseline_gospa': baseline_gospas,
        'baseline_false': baseline_false,
    }


def plot_training(rewards, gospas, eval_results, save_path):
    """Plot training curves and evaluation comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training reward
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.3, color='#4fc3f7')
    # Smoothed
    window = min(10, len(rewards) // 2)
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed, color='#4fc3f7', lw=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Reward')
    ax.grid(alpha=0.3)

    # Training GOSPA
    ax = axes[0, 1]
    ax.plot(gospas, alpha=0.3, color='#ff5252')
    if window > 1:
        smoothed = np.convolve(gospas, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(gospas)), smoothed, color='#ff5252', lw=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg GOSPA')
    ax.set_title('Training GOSPA (lower = better)')
    ax.grid(alpha=0.3)

    # Eval GOSPA comparison
    ax = axes[1, 0]
    data = [eval_results['baseline_gospa'], eval_results['rl_gospa']]
    bp = ax.boxplot(data, labels=['Fixed Threshold', 'RL Policy'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('#ff5252')
    bp['boxes'][1].set_facecolor('#4fc3f7')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_ylabel('Avg GOSPA')
    ax.set_title('Evaluation: GOSPA Distribution')
    ax.grid(alpha=0.3, axis='y')

    bl_mean = np.mean(eval_results['baseline_gospa'])
    rl_mean = np.mean(eval_results['rl_gospa'])
    improvement = (bl_mean - rl_mean) / bl_mean * 100
    ax.text(0.5, 0.95, f'Improvement: {improvement:+.1f}%',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=12, fontweight='bold',
            color='#69f0ae' if improvement > 0 else '#ff5252')

    # Eval False Tracks comparison
    ax = axes[1, 1]
    data = [eval_results['baseline_false'], eval_results['rl_false']]
    bp = ax.boxplot(data, labels=['Fixed Threshold', 'RL Policy'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('#ff5252')
    bp['boxes'][1].set_facecolor('#4fc3f7')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_ylabel('Total False Tracks')
    ax.set_title('Evaluation: False Track Count')
    ax.grid(alpha=0.3, axis='y')

    bl_f = np.mean(eval_results['baseline_false'])
    rl_f = np.mean(eval_results['rl_false'])
    ax.text(0.5, 0.95, f'False tracks: {bl_f:.0f} → {rl_f:.0f}',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=12, fontweight='bold')

    plt.suptitle('RL-based Track Management: Training & Evaluation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 70)
    print("RL-based Track Management for COP-DOA Multi-Target Tracking")
    print("=" * 70)

    # Train
    print("\n--- Training PPO Policy (100 episodes) ---\n")
    policy, rewards, gospas = train(n_episodes=100, n_scans=40, print_every=10)

    # Evaluate
    print("\n--- Evaluating: RL vs Fixed Threshold (20 scenarios) ---\n")
    eval_results = evaluate_comparison(policy, n_eval=20, n_scans=40)

    # Summary
    bl_gospa = np.mean(eval_results['baseline_gospa'])
    rl_gospa = np.mean(eval_results['rl_gospa'])
    bl_false = np.mean(eval_results['baseline_false'])
    rl_false = np.mean(eval_results['rl_false'])

    print(f"\n{'Metric':<25} {'Fixed Threshold':>15} {'RL Policy':>15} {'Change':>10}")
    print("-" * 65)
    print(f"{'Avg GOSPA':<25} {bl_gospa:>15.4f} {rl_gospa:>15.4f} "
          f"{(rl_gospa-bl_gospa)/bl_gospa*100:>+9.1f}%")
    print(f"{'Avg False Tracks':<25} {bl_false:>15.1f} {rl_false:>15.1f} "
          f"{(rl_false-bl_false)/max(bl_false,1)*100:>+9.1f}%")
    print("-" * 65)

    # Plot
    save_path = os.path.join(os.path.dirname(__file__), '..',
                             'results', 'figures', 'fig_rl_track_management.png')
    plot_training(rewards, gospas, eval_results, save_path)

    # Model size report
    print(f"\nPolicy: {policy.param_count()} params, "
          f"{policy.param_count()*4/1024:.1f}KB (fp32), "
          f"{policy.param_count()/1024:.1f}KB (INT8)")
    print("Edge deployment ready: fits within COP-RFS SRAM budget (~50KB)")


if __name__ == '__main__':
    main()
