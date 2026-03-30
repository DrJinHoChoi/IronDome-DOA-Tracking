#!/usr/bin/env python3
"""Experiment: COP vs T-COP (Temporal) vs T-COP+RL-Ready Tracker.

Compares three configurations on birth-death multi-target scenarios:
  1. Baseline COP-PHD: Standard COP + GM-PHD (scan-independent)
  2. T-COP-PHD: Temporal cumulant accumulation + tracker feedback
  3. T-COP-PHD + Adaptive: Time-series features for RL preparation

Metrics: GOSPA, Detection Rate, RMSE, False Track Rate, Track Switches.

Author: Jin Ho Choi
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import SubspaceCOP, TemporalCOP
from iron_dome_sim.tracking import COPPHD, ConstantVelocity
from iron_dome_sim.eval.metrics import gospa, rmse_doa, detection_rate


# ============================================================================
# Scenario: Birth-Death Multi-Target (time-varying K)
# ============================================================================
def birth_death_scenario(n_scans=60, M=8, snr_db=5, T=64):
    """Generate a birth-death scenario with crossing targets.

    Timeline:
      Scan  0-10: K=3 targets (static)
      Scan 10-20: K=5 (2 births)
      Scan 20-30: K=8 (3 births, 2 targets start crossing)
      Scan 30-40: K=6 (2 deaths, crossing resolved)
      Scan 40-50: K=4 (2 deaths)
      Scan 50-60: K=2 (2 deaths)

    Returns:
        scans: List of (X, true_doas) for each scan.
    """
    np.random.seed(42)
    ula = UniformLinearArray(M=M, d=0.5)

    # Define target trajectories (DOA in degrees over time)
    # Each target: (birth_scan, death_scan, doa_trajectory_func)
    targets = [
        # Core 3 targets (entire scenario)
        (0, 60, lambda t: -40 + 0.0 * t),       # Static at -40°
        (0, 60, lambda t: 20 + 0.3 * t),         # Slowly moving right
        (0, 40, lambda t: -10 - 0.2 * t),        # Moving left, dies at 40

        # Wave 1: births at scan 10
        (10, 50, lambda t: 50 - 0.5 * (t - 10)),  # Moving left from 50°
        (10, 45, lambda t: -60 + 0.4 * (t - 10)),  # Moving right from -60°

        # Wave 2: births at scan 20 (dense, crossing)
        (20, 55, lambda t: 5 + 0.8 * (t - 20)),    # Fast mover, crosses target[1]
        (20, 50, lambda t: -25 + 0.3 * (t - 20)),   # Slow mover
        (20, 35, lambda t: 70 - 1.0 * (t - 20)),    # Fast mover from right, short-lived
    ]

    scans = []
    for scan in range(n_scans):
        # Get active targets for this scan
        active_doas_deg = []
        for birth, death, traj in targets:
            if birth <= scan < death:
                doa_deg = traj(scan)
                if -85 <= doa_deg <= 85:  # Within FOV
                    active_doas_deg.append(doa_deg)

        true_doas = np.radians(np.array(sorted(active_doas_deg)))

        if len(true_doas) == 0:
            X = np.random.randn(M, T) + 1j * np.random.randn(M, T)
            X /= np.sqrt(2)
        else:
            X, _, _ = generate_snapshots(ula, true_doas, snr_db, T,
                                         signal_type="non_stationary")

        scans.append((X, true_doas))

    return scans, ula


# ============================================================================
# Run Tracker
# ============================================================================
def run_tracker(ula, cop_estimator, scans, use_physics=True):
    """Run COP-PHD tracker on a scenario.

    Returns per-scan results for evaluation.
    """
    motion = ConstantVelocity(dt=1.0, process_noise_std=np.radians(1.0))

    tracker = COPPHD(
        motion_model=motion,
        cop_estimator=cop_estimator,
        survival_prob=0.95,
        detection_prob=0.90,
        clutter_rate=2.0,
        birth_weight=0.1,
        prune_threshold=1e-5,
        merge_threshold=4.0,
        max_components=100,
        birth_pos_std_deg=2.0,
        birth_vel_std_deg=5.0,
        association_gate_deg=8.0,
        use_physics=use_physics,
    )

    # Coarser scan angles for speed (361 instead of 1801)
    scan_angles = np.linspace(-np.pi/2, np.pi/2, 361)

    results = []
    for scan_idx, (X, true_doas) in enumerate(scans):
        estimates, doa_meas, spectrum = tracker.process_scan(X, scan_angles=scan_angles)

        # Extract estimated DOAs
        est_doas = np.array([e[0][0] for e in estimates]) if estimates else np.array([])

        # Compute metrics
        if len(true_doas) > 0:
            est_pos = est_doas.reshape(-1, 1) if len(est_doas) > 0 else np.array([]).reshape(0, 1)
            true_pos = true_doas.reshape(-1, 1)
            gospa_val, gospa_decomp = gospa(est_pos, true_pos,
                                            c=np.radians(10), p=2, alpha=2)
            rmse_val, _ = rmse_doa(est_doas, true_doas)
            pd = detection_rate(est_doas, true_doas, threshold_rad=np.radians(5.0))
        else:
            gospa_val = 0.0
            gospa_decomp = {'localization': 0, 'missed': 0, 'false': 0}
            rmse_val = 0.0
            pd = 1.0 if len(est_doas) == 0 else 0.0

        n_false = max(0, len(est_doas) - len(true_doas))

        results.append({
            'scan': scan_idx,
            'true_doas': true_doas,
            'est_doas': est_doas,
            'doa_meas': doa_meas,
            'n_true': len(true_doas),
            'n_est': len(est_doas),
            'n_meas': len(doa_meas),
            'gospa': gospa_val,
            'gospa_loc': gospa_decomp['localization'],
            'gospa_miss': gospa_decomp['missed'],
            'gospa_false': gospa_decomp['false'],
            'rmse': rmse_val,
            'pd': pd,
            'n_false': n_false,
            'n_components': len(tracker.gm_components),
        })

    return results


# ============================================================================
# RL Environment Feature Extraction (preparation)
# ============================================================================
def extract_rl_features(tracker_results):
    """Extract per-scan features that an RL policy would observe.

    This is Phase 0: demonstrating what temporal features look like
    before building the full Gym environment.
    """
    features = []
    for i, r in enumerate(tracker_results):
        f = {
            'scan': i,
            # Track-level
            'n_tracks': r['n_est'],
            'n_measurements': r['n_meas'],
            'measurement_density': r['n_meas'] / np.pi,  # per radian
            # Temporal (requires history)
            'n_tracks_delta': r['n_est'] - tracker_results[i-1]['n_est'] if i > 0 else 0,
            'n_meas_delta': r['n_meas'] - tracker_results[i-1]['n_meas'] if i > 0 else 0,
            'gospa_delta': r['gospa'] - tracker_results[i-1]['gospa'] if i > 0 else 0,
            # Running averages
            'avg_gospa_5': np.mean([tracker_results[j]['gospa']
                                    for j in range(max(0,i-4), i+1)]),
            'avg_pd_5': np.mean([tracker_results[j]['pd']
                                 for j in range(max(0,i-4), i+1)]),
            # Component health
            'n_components': r['n_components'],
        }
        features.append(f)
    return features


# ============================================================================
# Visualization
# ============================================================================
def plot_results(all_results, labels, save_path=None):
    """Plot comparison of multiple tracker configurations."""
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

    colors = ['#ff5252', '#4fc3f7', '#69f0ae', '#ffd740']

    # 1. True vs Estimated target count
    ax1 = fig.add_subplot(gs[0, 0])
    scans_x = [r['scan'] for r in all_results[0]]
    n_true = [r['n_true'] for r in all_results[0]]
    ax1.plot(scans_x, n_true, 'k--', lw=2, label='True K', alpha=0.7)
    for i, (results, label) in enumerate(zip(all_results, labels)):
        n_est = [r['n_est'] for r in results]
        ax1.plot(scans_x, n_est, color=colors[i], lw=1.5, label=label, alpha=0.9)
    ax1.set_xlabel('Scan')
    ax1.set_ylabel('Number of Targets')
    ax1.set_title('Target Count: True vs Estimated')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # 2. GOSPA over time
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (results, label) in enumerate(zip(all_results, labels)):
        gospa_vals = [r['gospa'] for r in results]
        ax2.plot(scans_x, gospa_vals, color=colors[i], lw=1.5, label=label)
    ax2.set_xlabel('Scan')
    ax2.set_ylabel('GOSPA')
    ax2.set_title('GOSPA Metric (lower = better)')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # 3. Detection Rate
    ax3 = fig.add_subplot(gs[1, 0])
    for i, (results, label) in enumerate(zip(all_results, labels)):
        pd_vals = [r['pd'] for r in results]
        ax3.plot(scans_x, pd_vals, color=colors[i], lw=1.5, label=label)
    ax3.set_xlabel('Scan')
    ax3.set_ylabel('Detection Rate')
    ax3.set_title('Detection Rate (higher = better)')
    ax3.set_ylim(-0.05, 1.15)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # 4. RMSE
    ax4 = fig.add_subplot(gs[1, 1])
    for i, (results, label) in enumerate(zip(all_results, labels)):
        rmse_vals = [np.degrees(r['rmse']) if r['rmse'] < 100 else np.nan for r in results]
        ax4.plot(scans_x, rmse_vals, color=colors[i], lw=1.5, label=label)
    ax4.set_xlabel('Scan')
    ax4.set_ylabel('RMSE (degrees)')
    ax4.set_title('DOA RMSE')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # 5. GOSPA Decomposition (stacked bar) for best method
    ax5 = fig.add_subplot(gs[2, 0])
    best_idx = np.argmin([np.mean([r['gospa'] for r in res]) for res in all_results])
    best_results = all_results[best_idx]
    loc_vals = [r['gospa_loc'] for r in best_results]
    miss_vals = [r['gospa_miss'] for r in best_results]
    false_vals = [r['gospa_false'] for r in best_results]
    ax5.fill_between(scans_x, 0, loc_vals, alpha=0.6, color='#4fc3f7', label='Localization')
    bottom = np.array(loc_vals)
    ax5.fill_between(scans_x, bottom, bottom + np.array(miss_vals), alpha=0.6,
                     color='#ff9100', label='Missed')
    bottom2 = bottom + np.array(miss_vals)
    ax5.fill_between(scans_x, bottom2, bottom2 + np.array(false_vals), alpha=0.6,
                     color='#ff5252', label='False')
    ax5.set_xlabel('Scan')
    ax5.set_ylabel('GOSPA Component')
    ax5.set_title(f'GOSPA Decomposition ({labels[best_idx]})')
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    # 6. DOA tracks over time
    ax6 = fig.add_subplot(gs[2, 1])
    for r in all_results[0]:
        for doa in r['true_doas']:
            ax6.plot(r['scan'], np.degrees(doa), 'k.', ms=2, alpha=0.3)
    best_results = all_results[best_idx]
    for r in best_results:
        for doa in r['est_doas']:
            ax6.plot(r['scan'], np.degrees(doa), '.', color=colors[best_idx], ms=3, alpha=0.5)
    ax6.set_xlabel('Scan')
    ax6.set_ylabel('DOA (degrees)')
    ax6.set_title(f'DOA Tracks ({labels[best_idx]})')
    ax6.grid(alpha=0.3)

    # 7. RL Feature Preview: temporal deltas
    ax7 = fig.add_subplot(gs[3, 0])
    rl_feats = extract_rl_features(all_results[best_idx])
    ax7.plot(scans_x, [f['n_tracks_delta'] for f in rl_feats],
             color='#7c4dff', lw=1.5, label='Track count delta')
    ax7.plot(scans_x, [f['n_meas_delta'] for f in rl_feats],
             color='#ff9100', lw=1.5, label='Measurement delta', alpha=0.7)
    ax7.axhline(0, color='gray', ls='--', alpha=0.5)
    ax7.set_xlabel('Scan')
    ax7.set_ylabel('Delta')
    ax7.set_title('RL State Features: Temporal Deltas')
    ax7.legend(fontsize=8)
    ax7.grid(alpha=0.3)

    # 8. Summary bar chart
    ax8 = fig.add_subplot(gs[3, 1])
    metrics_names = ['Avg GOSPA', 'Avg Pd', 'Avg RMSE (°)']
    x_bar = np.arange(len(metrics_names))
    width = 0.25
    for i, (results, label) in enumerate(zip(all_results, labels)):
        avg_gospa = np.mean([r['gospa'] for r in results])
        avg_pd = np.mean([r['pd'] for r in results])
        avg_rmse = np.mean([np.degrees(r['rmse']) for r in results if r['rmse'] < 100])
        vals = [avg_gospa, avg_pd, avg_rmse]
        ax8.bar(x_bar + i * width, vals, width, color=colors[i], label=label, alpha=0.85)
    ax8.set_xticks(x_bar + width)
    ax8.set_xticklabels(metrics_names, fontsize=9)
    ax8.set_title('Summary Comparison')
    ax8.legend(fontsize=8)
    ax8.grid(alpha=0.3, axis='y')

    plt.suptitle('COP vs T-COP: Temporal Accumulation Impact on Multi-Target Tracking',
                 fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("COP vs T-COP: Temporal Accumulation for Multi-Target Tracking")
    print("=" * 70)

    # Generate scenario
    M = 8
    SNR = 5  # Moderate SNR to see temporal benefit
    T = 64
    N_SCANS = 40

    print(f"\nScenario: M={M}, SNR={SNR}dB, T={T}, {N_SCANS} scans")
    print("Birth-death with crossing targets (K varies 2-8)")
    print()

    scans, ula = birth_death_scenario(n_scans=N_SCANS, M=M, snr_db=SNR, T=T)

    # Configuration 1: Baseline COP-PHD
    print("[1/3] Running Baseline COP-PHD...")
    cop_baseline = SubspaceCOP(ula, rho=2)
    results_baseline = run_tracker(ula, cop_baseline, scans, use_physics=True)

    # Configuration 2: T-COP-PHD (temporal, alpha=0.85)
    print("[2/3] Running T-COP-PHD (alpha=0.85)...")
    tcop = TemporalCOP(ula, rho=2, alpha=0.85, prior_weight=0.3,
                        search_width_deg=15.0)
    results_tcop = run_tracker(ula, tcop, scans, use_physics=True)

    # Configuration 3: T-COP-PHD (aggressive temporal, alpha=0.70)
    print("[3/3] Running T-COP-PHD (alpha=0.70, aggressive)...")
    tcop_agg = TemporalCOP(ula, rho=2, alpha=0.70, prior_weight=0.5,
                            search_width_deg=12.0)
    results_tcop_agg = run_tracker(ula, tcop_agg, scans, use_physics=True)

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {'Baseline COP':>15} {'T-COP (0.85)':>15} {'T-COP (0.70)':>15}")
    print("-" * 70)

    for name, key, fmt, invert in [
        ('Avg GOSPA', 'gospa', '.4f', False),
        ('Avg Detection Rate', 'pd', '.3f', True),
        ('Avg RMSE (deg)', 'rmse', '.2f', False),
        ('Total False Tracks', 'n_false', 'd', False),
    ]:
        vals = []
        for results in [results_baseline, results_tcop, results_tcop_agg]:
            if key == 'rmse':
                v = np.mean([np.degrees(r[key]) for r in results if r[key] < 100])
            elif key == 'n_false':
                v = sum(r[key] for r in results)
            else:
                v = np.mean([r[key] for r in results])
            vals.append(v)

        best_idx = np.argmin(vals) if not invert else np.argmax(vals)
        row = f"{name:<25}"
        for i, v in enumerate(vals):
            marker = " *" if i == best_idx else "  "
            if fmt == 'd':
                row += f"{int(v):>13}{marker}"
            else:
                row += f"{v:>13{fmt}}{marker}"
        print(row)

    print("-" * 70)
    print("* = best\n")

    # RL feature summary
    rl_feats = extract_rl_features(results_tcop)
    print("RL State Feature Statistics (T-COP):")
    print(f"  Track count delta  : mean={np.mean([f['n_tracks_delta'] for f in rl_feats]):.2f}, "
          f"std={np.std([f['n_tracks_delta'] for f in rl_feats]):.2f}")
    print(f"  Measurement delta  : mean={np.mean([f['n_meas_delta'] for f in rl_feats]):.2f}, "
          f"std={np.std([f['n_meas_delta'] for f in rl_feats]):.2f}")
    print(f"  Avg GOSPA (5-scan) : mean={np.mean([f['avg_gospa_5'] for f in rl_feats]):.4f}")
    print(f"  Avg Pd (5-scan)    : mean={np.mean([f['avg_pd_5'] for f in rl_feats]):.3f}")
    print(f"  Component count    : mean={np.mean([f['n_components'] for f in rl_feats]):.1f}, "
          f"max={max(f['n_components'] for f in rl_feats)}")

    # Plot
    save_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                             'figures', 'fig_temporal_cop_comparison.png')
    plot_results(
        [results_baseline, results_tcop, results_tcop_agg],
        ['Baseline COP', 'T-COP (α=0.85)', 'T-COP (α=0.70)'],
        save_path=save_path,
    )


if __name__ == '__main__':
    main()
