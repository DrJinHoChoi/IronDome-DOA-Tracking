#!/usr/bin/env python3
"""Train Mamba-COP-RL on combat-realistic scenarios.

This experiment is designed to show Mamba's advantage over MLP
when the observation is high-dimensional and sequences are long:

  obs_dim : 183  (raw COP spectrum 181-dim + 2 global features)
  n_scans : 200  (long episodes with temporal dependencies)
  scenarios: Jamming / Stealth / Saturation / Formation

Expected result: Mamba > MLP on Stealth + Jamming (long-range memory),
                 similar on Saturation + Formation (short-range reaction).

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

from iron_dome_sim.rl.combat_env import CombatTrackEnv, SCENARIO_TYPES
from iron_dome_sim.rl.policy import TrackPolicy, PPOTrainer, RolloutBuffer
from iron_dome_sim.rl.mamba_encoder import MambaCOPPolicy


OBS_DIM = CombatTrackEnv.OBS_DIM   # 183
ACT_DIM = CombatTrackEnv.ACT_DIM   # 3
N_SCANS = 200
N_TRAIN = 300
N_EVAL  = 20


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_policy(policy_cls, policy_kwargs, n_episodes=N_TRAIN,
                 n_scans=N_SCANS, lr=3e-4, entropy_coef=0.02,
                 label="Policy"):
    policy = policy_cls(**policy_kwargs)
    trainer = PPOTrainer(policy, lr=lr, clip_ratio=0.2, epochs=4,
                         entropy_coef=entropy_coef)
    buffer = RolloutBuffer()

    print(f"\n[{label}] Params: {policy.param_count()}, "
          f"Size: {policy.param_count()/1024:.1f}KB INT8")

    best_gospa = float('inf')
    episode_gospas = []

    for ep in range(n_episodes):
        snr = np.random.uniform(5, 15)
        env = CombatTrackEnv(M=8, snr_db=snr, T=64, n_scans=n_scans)
        obs = env.reset(seed=ep)

        if hasattr(policy, 'reset_hidden'):
            policy.reset_hidden()

        ep_reward = 0
        ep_infos  = []

        for _ in range(n_scans):
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

        avg_gospa = np.mean([i['gospa'] for i in ep_infos])
        episode_gospas.append(avg_gospa)
        if avg_gospa < best_gospa:
            best_gospa = avg_gospa

        if (ep + 1) % 50 == 0:
            recent = np.mean(episode_gospas[-50:])
            print(f"  Ep {ep+1:4d} | GOSPA: {recent:.4f} | "
                  f"Best: {best_gospa:.4f} | PL: {losses['policy_loss']:.4f}")

    return policy, episode_gospas


# ------------------------------------------------------------------
# Evaluation — per scenario type
# ------------------------------------------------------------------

def evaluate_per_scenario(policies, labels, n_eval=N_EVAL, n_scans=N_SCANS):
    """Evaluate each policy on each scenario type separately."""
    results = {stype: {label: [] for label in labels}
               for stype in SCENARIO_TYPES}

    for stype in SCENARIO_TYPES:
        print(f"  Evaluating: {stype} ...")
        for seed in range(n_eval):
            snr = np.random.RandomState(3000 + seed).uniform(5, 15)

            for policy, label in zip(policies, labels):
                env = CombatTrackEnv(M=8, snr_db=snr, T=64,
                                     n_scans=n_scans,
                                     scenario_type=stype)
                obs = env.reset(seed=3000 + seed)

                if hasattr(policy, 'reset_hidden'):
                    policy.reset_hidden()

                ep_gospa = []
                for _ in range(n_scans):
                    if policy is None:
                        action = np.zeros(ACT_DIM)
                    else:
                        action, _, _ = policy.get_action(obs,
                                                         deterministic=True)
                    obs, _, done, info = env.step(action)
                    ep_gospa.append(info['gospa'])
                    if done:
                        break

                results[stype][label].append(np.mean(ep_gospa))

    return results


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_results(train_gospas, eval_results, labels, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {'Fixed':     '#ff5252',
              'MLP (PPO)': '#4fc3f7',
              'Mamba-COP': '#69f0ae'}

    # Training curves
    ax = axes[0, 0]
    for label, gospas in train_gospas.items():
        c = colors.get(label, '#888')
        ax.plot(gospas, alpha=0.15, color=c)
        w = min(20, len(gospas) // 3)
        if w > 1:
            sm = np.convolve(gospas, np.ones(w) / w, mode='valid')
            ax.plot(range(w - 1, len(gospas)), sm, color=c, lw=2,
                    label=label)
    ax.set_title('Training GOSPA (all scenarios mixed)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg GOSPA')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Per-scenario bar chart
    ax = axes[0, 1]
    x = np.arange(len(SCENARIO_TYPES))
    w = 0.25
    for i, label in enumerate(labels):
        means = [np.mean(eval_results[s][label]) for s in SCENARIO_TYPES]
        ax.bar(x + (i - 1) * w, means, w,
               color=colors.get(label, '#888'), label=label, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_TYPES, fontsize=9)
    ax.set_ylabel('Avg GOSPA')
    ax.set_title('GOSPA per Scenario (lower = better)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # Per-scenario improvement: Mamba vs MLP
    ax = axes[0, 2]
    if 'MLP (PPO)' in labels and 'Mamba-COP' in labels:
        improv = []
        for s in SCENARIO_TYPES:
            mlp_g   = np.mean(eval_results[s]['MLP (PPO)'])
            mamba_g = np.mean(eval_results[s]['Mamba-COP'])
            improv.append((mlp_g - mamba_g) / (mlp_g + 1e-8) * 100)
        bars = ax.bar(SCENARIO_TYPES, improv,
                      color=['#69f0ae' if v > 0 else '#ff5252'
                             for v in improv], alpha=0.85)
        ax.axhline(0, color='white', lw=1, ls='--')
        ax.set_ylabel('Improvement over MLP (%)')
        ax.set_title('Mamba vs MLP: % GOSPA improvement')
        for bar, v in zip(bars, improv):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.3 if v >= 0 else -0.8),
                    f'{v:+.1f}%', ha='center', fontsize=10,
                    fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

    # Boxplots per scenario (Mamba vs MLP)
    for idx, stype in enumerate(SCENARIO_TYPES):
        ax = axes[1, idx % 3] if len(SCENARIO_TYPES) <= 3 else axes[1, idx - 1]
        if idx >= 3:
            break
        data = [eval_results[stype][l] for l in labels
                if l in eval_results[stype]]
        bp = ax.boxplot(data,
                        tick_labels=[l[:10] for l in labels
                                     if l in eval_results[stype]],
                        patch_artist=True)
        for j, box in enumerate(bp['boxes']):
            box.set_facecolor(list(colors.values())[j])
            box.set_alpha(0.7)
        ax.set_title(f'{stype.capitalize()} Scenario')
        ax.set_ylabel('Avg GOSPA')
        ax.grid(alpha=0.3, axis='y')

    # 4th scenario in spare cell
    ax = axes[1, 2]
    stype = SCENARIO_TYPES[3]
    data = [eval_results[stype][l] for l in labels
            if l in eval_results[stype]]
    bp = ax.boxplot(data,
                    tick_labels=[l[:10] for l in labels
                                 if l in eval_results[stype]],
                    patch_artist=True)
    for j, box in enumerate(bp['boxes']):
        box.set_facecolor(list(colors.values())[j])
        box.set_alpha(0.7)
    ax.set_title(f'{stype.capitalize()} Scenario')
    ax.set_ylabel('Avg GOSPA')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle(
        'Mamba-COP-RL: Combat Scenarios\n'
        f'obs_dim={OBS_DIM} (raw spectrum), n_scans={N_SCANS}',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Mamba-COP-RL: Combat Environment Training")
    print(f"  obs_dim={OBS_DIM} (raw COP spectrum), n_scans={N_SCANS}")
    print(f"  Scenarios: {SCENARIO_TYPES}")
    print("=" * 70)

    # 1. Train MLP baseline
    print("\n[1/2] Training MLP (PPO) ...")
    mlp_policy, mlp_gospas = train_policy(
        TrackPolicy,
        {'obs_dim': OBS_DIM, 'act_dim': ACT_DIM, 'hidden': 64},
        lr=3e-4, entropy_coef=0.02,
        label="MLP (PPO)"
    )

    # 2. Train Mamba-COP-RL
    # d_hidden=48, d_state=24 → ~41K params (fits 50KB SRAM)
    print("\n[2/2] Training Mamba-COP-RL ...")
    mamba_policy, mamba_gospas = train_policy(
        MambaCOPPolicy,
        {'obs_dim': OBS_DIM, 'act_dim': ACT_DIM,
         'd_state': 24, 'd_hidden': 48, 'policy_hidden': 64},
        lr=3e-4, entropy_coef=0.02,
        label="Mamba-COP"
    )

    # 3. Evaluate per scenario
    print(f"\n--- Evaluating {N_EVAL} scenarios per type ---")
    labels  = ['Fixed', 'MLP (PPO)', 'Mamba-COP']
    policies = [None, mlp_policy, mamba_policy]
    eval_results = evaluate_per_scenario(policies, labels,
                                         n_eval=N_EVAL, n_scans=N_SCANS)

    # 4. Summary table
    print(f"\n{'='*72}")
    print(f"{'Scenario':<14}", end='')
    for l in labels:
        print(f"  {l:>12}", end='')
    print(f"  {'Mamba vs MLP':>14}")
    print('-' * 72)
    for stype in SCENARIO_TYPES:
        print(f"{stype:<14}", end='')
        vals = []
        for l in labels:
            v = np.mean(eval_results[stype][l])
            vals.append(v)
            print(f"  {v:>12.4f}", end='')
        mlp_v   = vals[labels.index('MLP (PPO)')]
        mamba_v = vals[labels.index('Mamba-COP')]
        pct = (mlp_v - mamba_v) / (mlp_v + 1e-8) * 100
        print(f"  {pct:>+13.1f}%")
    print('=' * 72)

    print(f"\nMLP   : {mlp_policy.param_count()} params, "
          f"{mlp_policy.param_count()/1024:.1f}KB INT8")
    print(f"Mamba : {mamba_policy.param_count()} params, "
          f"{mamba_policy.param_count()/1024:.1f}KB INT8")

    # 5. Save model
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    torch.save(mamba_policy.state_dict(),
               os.path.join(save_dir, 'mamba_combat_best.pt'))

    # 6. Plot
    fig_path = os.path.join(save_dir, 'figures', 'fig_combat_mamba.png')
    plot_results(
        {'MLP (PPO)': mlp_gospas, 'Mamba-COP': mamba_gospas},
        eval_results, labels, fig_path
    )


if __name__ == '__main__':
    main()
