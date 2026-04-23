#!/usr/bin/env python3
"""Tracking Ablation Study: Physics-based GM-PHD vs Standard GM-PHD.

Isolates the contribution of the physics-based tracking innovations:
  1. Hungarian measurement-to-track identification BEFORE update
  2. Velocity-gated merge to preserve tracks through crossings
  3. Birth from unassociated-only measurements

Comparison (2x2 factorial):
  A. COP + Physics-based GM-PHD  (PROPOSED)
  B. COP + Standard GM-PHD       (ablation: tracking only)
  C. MUSIC + Physics-based GM-PHD (ablation: estimation only)
  D. MUSIC + Standard GM-PHD     (conventional baseline)

Key scenario: 4 crossing targets — crossing is the critical test for tracking.

Output: results/figures/fig_tracking_ablation.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import SubspaceCOP, MUSIC
from iron_dome_sim.tracking import COPPHD, ConstantVelocity
from iron_dome_sim.eval.metrics import rmse_doa, detection_rate, gospa

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 22,
    'axes.labelsize': 25,
    'axes.titlesize': 25,
    'legend.fontsize': 17,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'lines.linewidth': 2.5,
    'lines.markersize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 1.2,
})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Parameters ──────────────────────────────────────────────────────────
M = 8
K = 4  # 4 crossing targets (determined: K < M-1)
SNR_DB = 15
T = 512
N_SCANS = 30
DT = 1.0
SCAN_ANGLES = np.linspace(-np.pi / 2, np.pi / 2, 1801)

# 4 targets with crossing trajectories
# Target 1: starts at -40°, moves right at +2.5°/scan
# Target 2: starts at +40°, moves left at -2.5°/scan
# Target 3: starts at -20°, moves right at +1.5°/scan
# Target 4: starts at +20°, moves left at -1.5°/scan
# Crossings occur around scans ~8 and ~16
BASE_DOAS = np.radians([-40, 40, -20, 20])
RATES = np.radians([2.5, -2.5, 1.5, -1.5])

TARGET_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
TARGET_MARKERS = ['o', 's', 'D', '^']


def build_phd(estimator, use_physics=True):
    """Construct a COPPHD tracker."""
    model = ConstantVelocity(dt=DT, process_noise_std=np.radians(0.5))
    return COPPHD(
        model, estimator,
        survival_prob=0.95,
        detection_prob=0.95,
        birth_weight=0.5,
        clutter_rate=0.3,
        prune_threshold=1e-3,
        merge_threshold=2.0,
        birth_pos_std_deg=3.0,
        birth_vel_std_deg=5.0,
        association_gate_deg=10.0,
        use_physics=use_physics,
    )


def run_pipeline(label, estimator, use_physics=True):
    """Run full tracking pipeline and collect per-scan metrics."""
    array = UniformLinearArray(M=M, d=0.5)
    phd = build_phd(estimator, use_physics=use_physics)

    true_all = []
    est_all = []
    label_hist = []
    pd_per_scan = []
    rmse_per_scan = []
    gospa_per_scan = []

    for si in range(N_SCANS):
        true_doas = BASE_DOAS + RATES * si
        true_doas = np.clip(true_doas, -np.pi/2 + 0.05, np.pi/2 - 0.05)

        np.random.seed(42 + si)
        X, _, _ = generate_snapshots(array, true_doas, SNR_DB, T, "non_stationary")

        phd.process_scan(X, SCAN_ANGLES)
        est_doas = phd.get_doa_estimates()
        track_states = phd.get_track_states()

        true_all.append(true_doas)
        est_all.append(est_doas)
        label_hist.append(track_states)

        # Detection rate (3-degree threshold)
        pd, _ = detection_rate(est_doas, true_doas, np.radians(3.0))
        pd_per_scan.append(pd)

        # RMSE (only matched)
        if len(est_doas) > 0:
            r, _ = rmse_doa(est_doas, true_doas)
            rmse_per_scan.append(np.degrees(r))
        else:
            rmse_per_scan.append(90.0)

        # GOSPA
        g, decomp = gospa(
            est_doas.reshape(-1, 1) if len(est_doas) > 0 else np.empty((0, 1)),
            true_doas.reshape(-1, 1),
            c=np.radians(10.0), p=2, alpha=2,
        )
        gospa_per_scan.append(np.degrees(g))

    return dict(
        true=true_all, est=est_all, labels=label_hist,
        pd=pd_per_scan, rmse=rmse_per_scan, gospa=gospa_per_scan,
    )


def assign_labels(label_hist, K, base_doas, rates):
    """Map PHD labels to source indices."""
    all_labels = set()
    for tlh in label_hist:
        all_labels.update(tlh.keys())

    l2s = {}
    for label in sorted(all_labels):
        states, sids = [], []
        for si, tlh in enumerate(label_hist):
            if label in tlh:
                states.append(tlh[label][0])
                sids.append(si)
        if not states:
            continue
        best_k, best_c = -1, float('inf')
        for k in range(K):
            c = sum(
                abs(s[0] - np.clip(base_doas[k] + rates[k]*si,
                                    -np.pi/2+0.05, np.pi/2-0.05))
                + 2.0 * (abs(s[2] - rates[k]) if len(s) > 2 else 0)
                for s, si in zip(states, sids)
            ) / len(states)
            if c < best_c:
                best_c, best_k = c, k
        l2s[label] = best_k
    return l2s


def count_track_switches(label_hist, l2s, K, N_SCANS):
    """Count how many times a track switches its identity assignment."""
    # For each scan, check which label is assigned to which source
    source_to_labels = {k: [] for k in range(K)}
    for si, tlh in enumerate(label_hist):
        for label, (state, _, _) in tlh.items():
            src = l2s.get(label, -1)
            if 0 <= src < K:
                source_to_labels[src].append((si, label))

    switches = 0
    for k in range(K):
        labels = source_to_labels[k]
        for i in range(1, len(labels)):
            if labels[i][1] != labels[i-1][1]:
                switches += 1
    return switches


def plot_tracking(ax, true_all, label_hist, l2s, K, title):
    """Plot DOA tracking panel.

    Color convention:
      - True DOAs: per-target colored solid lines + GREEN circles
      - Estimated DOAs: RED upward triangles with black edge
    """
    scans = np.arange(1, N_SCANS + 1)

    # True tracks: colored solid lines + green circles at each scan
    for k in range(K):
        doas = [np.degrees(true_all[i][k]) for i in range(N_SCANS)]
        ax.plot(scans, doas, '-', color=TARGET_COLORS[k % len(TARGET_COLORS)],
                linewidth=3.0, alpha=0.4, zorder=2)
        ax.plot(scans, doas, 'o', color='#2ca02c', markersize=6,
                markeredgecolor='darkgreen', markeredgewidth=0.5,
                alpha=0.5, zorder=3)

    # Estimated DOAs: ALL red triangles (clearly distinct from green true)
    for si, tlh in enumerate(label_hist):
        for label, (state, _, _) in tlh.items():
            ax.plot(si + 1, np.degrees(state[0]), '^', color='#d62728',
                    markersize=9, markeredgecolor='black', markeredgewidth=0.8,
                    alpha=0.85, zorder=5)

    ax.set_ylabel('DOA (degrees)')
    ax.set_xlabel('Scan')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim([0.5, N_SCANS + 0.5])


def main():
    print("=" * 70)
    print("Tracking Ablation: Physics-based GM-PHD vs Standard GM-PHD")
    print(f"  M={M}, K={K} crossing targets, SNR={SNR_DB} dB, T={T}, {N_SCANS} scans")
    print("=" * 70)

    array = UniformLinearArray(M=M, d=0.5)

    # A. COP + Physics-based GM-PHD (PROPOSED)
    print("\n[A] COP + Physics-based GM-PHD (PROPOSED)...")
    cop_a = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    res_A = run_pipeline("COP+Physics", cop_a, use_physics=True)

    # B. COP + Standard GM-PHD (ablation: tracking only)
    print("[B] COP + Standard GM-PHD (ablation)...")
    cop_b = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    res_B = run_pipeline("COP+Standard", cop_b, use_physics=False)

    # C. MUSIC + Physics-based GM-PHD
    print("[C] MUSIC + Physics-based GM-PHD...")
    music_c = MUSIC(array, num_sources=K)
    res_C = run_pipeline("MUSIC+Physics", music_c, use_physics=True)

    # D. MUSIC + Standard GM-PHD (conventional baseline)
    print("[D] MUSIC + Standard GM-PHD (conventional)...")
    music_d = MUSIC(array, num_sources=K)
    res_D = run_pipeline("MUSIC+Standard", music_d, use_physics=False)

    # Label assignments
    l2s_A = assign_labels(res_A['labels'], K, BASE_DOAS, RATES)
    l2s_B = assign_labels(res_B['labels'], K, BASE_DOAS, RATES)
    l2s_C = assign_labels(res_C['labels'], K, BASE_DOAS, RATES)
    l2s_D = assign_labels(res_D['labels'], K, BASE_DOAS, RATES)

    # Track switches
    sw_A = count_track_switches(res_A['labels'], l2s_A, K, N_SCANS)
    sw_B = count_track_switches(res_B['labels'], l2s_B, K, N_SCANS)
    sw_C = count_track_switches(res_C['labels'], l2s_C, K, N_SCANS)
    sw_D = count_track_switches(res_D['labels'], l2s_D, K, N_SCANS)

    # ── Summary ─────────────────────────────────────────────────────
    warmup = 3
    results = {
        'A (COP+Physics)': res_A,
        'B (COP+Standard)': res_B,
        'C (MUSIC+Physics)': res_C,
        'D (MUSIC+Standard)': res_D,
    }
    switches = {
        'A (COP+Physics)': sw_A,
        'B (COP+Standard)': sw_B,
        'C (MUSIC+Physics)': sw_C,
        'D (MUSIC+Standard)': sw_D,
    }

    print(f"\n{'Pipeline':<25s} {'Pd(%)':>8s} {'RMSE(°)':>8s} {'GOSPA(°)':>9s} {'Switches':>9s}")
    print("-" * 65)
    for name, res in results.items():
        avg_pd = np.mean(res['pd'][warmup:]) * 100
        avg_rmse = np.mean(res['rmse'][warmup:])
        avg_gospa = np.mean(res['gospa'][warmup:])
        sw = switches[name]
        print(f"{name:<25s} {avg_pd:>7.1f}% {avg_rmse:>8.2f} {avg_gospa:>9.2f} {sw:>9d}")

    # ── Improvement calculation ──
    avg_pd_A = np.mean(res_A['pd'][warmup:])
    avg_pd_B = np.mean(res_B['pd'][warmup:])
    avg_pd_D = np.mean(res_D['pd'][warmup:])
    avg_rmse_A = np.mean(res_A['rmse'][warmup:])
    avg_rmse_B = np.mean(res_B['rmse'][warmup:])
    avg_rmse_D = np.mean(res_D['rmse'][warmup:])
    avg_gospa_A = np.mean(res_A['gospa'][warmup:])
    avg_gospa_B = np.mean(res_B['gospa'][warmup:])
    avg_gospa_D = np.mean(res_D['gospa'][warmup:])

    print(f"\n=== Tracking Improvement (Physics vs Standard, same COP front-end) ===")
    print(f"  Pd:    {avg_pd_A*100:.1f}% vs {avg_pd_B*100:.1f}%  (Δ = +{(avg_pd_A-avg_pd_B)*100:.1f}%p)")
    print(f"  RMSE:  {avg_rmse_A:.2f}° vs {avg_rmse_B:.2f}°  (Δ = {avg_rmse_A-avg_rmse_B:+.2f}°)")
    print(f"  GOSPA: {avg_gospa_A:.2f}° vs {avg_gospa_B:.2f}°  (Δ = {avg_gospa_A-avg_gospa_B:+.2f}°)")
    print(f"  Track Switches: {sw_A} vs {sw_B}  (Δ = {sw_A-sw_B:+d})")

    print(f"\n=== Full System Improvement (COP+Physics vs MUSIC+Standard) ===")
    print(f"  Pd:    {avg_pd_A*100:.1f}% vs {avg_pd_D*100:.1f}%  (Δ = +{(avg_pd_A-avg_pd_D)*100:.1f}%p)")
    print(f"  RMSE:  {avg_rmse_A:.2f}° vs {avg_rmse_D:.2f}°  (Δ = {avg_rmse_A-avg_rmse_D:+.2f}°)")
    print(f"  GOSPA: {avg_gospa_A:.2f}° vs {avg_gospa_D:.2f}°  (Δ = {avg_gospa_A-avg_gospa_D:+.2f}°)")

    # ── Crossing-specific analysis ──
    # Crossings happen around scans 8 and 16
    crossing_scans = list(range(6, 12)) + list(range(14, 20))  # scans 7-12, 15-20
    non_crossing_scans = [s for s in range(warmup, N_SCANS) if s not in crossing_scans]

    print(f"\n=== Crossing vs Non-Crossing Performance (COP front-end) ===")
    for label, res, name in [
        ('A', res_A, 'Physics'), ('B', res_B, 'Standard')]:
        crossing_pd = np.mean([res['pd'][s] for s in crossing_scans]) * 100
        non_crossing_pd = np.mean([res['pd'][s] for s in non_crossing_scans]) * 100
        crossing_rmse = np.mean([res['rmse'][s] for s in crossing_scans])
        non_crossing_rmse = np.mean([res['rmse'][s] for s in non_crossing_scans])
        print(f"  {name:<10s}: Crossing Pd={crossing_pd:.1f}% RMSE={crossing_rmse:.2f}°  |  "
              f"Non-crossing Pd={non_crossing_pd:.1f}% RMSE={non_crossing_rmse:.2f}°")

    # ── Figure: 3x2 layout ─────────────────────────────────────────
    fig = plt.figure(figsize=(22, 24))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25,
                          height_ratios=[3, 3, 2])

    # (a) COP + Physics-based (PROPOSED)
    ax_a = fig.add_subplot(gs[0, 0])
    plot_tracking(ax_a, res_A['true'], res_A['labels'], l2s_A, K,
                  f'(a) COP + Physics-based GM-PHD (Proposed)')
    leg = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markeredgecolor='darkgreen', markersize=9, label='True DOA'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#d62728',
               markeredgecolor='black', markersize=9, label='Estimated DOA'),
    ]
    ax_a.legend(handles=leg, loc='upper right', fontsize=15)

    # (b) COP + Standard (ablation)
    ax_b = fig.add_subplot(gs[0, 1])
    plot_tracking(ax_b, res_B['true'], res_B['labels'], l2s_B, K,
                  f'(b) COP + Standard GM-PHD (No identification)')
    ax_b.legend(handles=leg, loc='upper right', fontsize=15)

    # Mark crossing regions
    for ax in [ax_a, ax_b]:
        ax.axvspan(7, 11, alpha=0.1, color='red', label='Crossing zone')
        ax.axvspan(15, 19, alpha=0.1, color='red')

    # (c) MUSIC + Physics-based
    ax_c = fig.add_subplot(gs[1, 0])
    plot_tracking(ax_c, res_C['true'], res_C['labels'], l2s_C, K,
                  f'(c) MUSIC + Physics-based GM-PHD')
    leg_cd = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markeredgecolor='darkgreen', markersize=9, label='True DOA'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#d62728',
               markeredgecolor='black', markersize=9, label='Estimated DOA'),
    ]
    ax_c.legend(handles=leg_cd, loc='upper right', fontsize=15)

    # (d) MUSIC + Standard (conventional baseline)
    ax_d = fig.add_subplot(gs[1, 1])
    plot_tracking(ax_d, res_D['true'], res_D['labels'], l2s_D, K,
                  f'(d) MUSIC + Standard GM-PHD (Conventional)')
    ax_d.legend(handles=leg_cd, loc='upper right', fontsize=15)

    for ax in [ax_c, ax_d]:
        ax.axvspan(7, 11, alpha=0.1, color='red')
        ax.axvspan(15, 19, alpha=0.1, color='red')

    # (e) GOSPA comparison
    scans = np.arange(1, N_SCANS + 1)
    ax_gospa = fig.add_subplot(gs[2, 0])
    ax_gospa.plot(scans, res_A['gospa'], '-^', color='#0055CC',
                  markersize=6, linewidth=2.0, label='COP+Physics (Proposed)')
    ax_gospa.plot(scans, res_B['gospa'], '-s', color='#CC0000',
                  markersize=6, linewidth=2.0, label='COP+Standard (Ablation)')
    ax_gospa.plot(scans, res_C['gospa'], '--^', color='#0055CC',
                  markersize=5, linewidth=1.5, alpha=0.6, label='MUSIC+Physics')
    ax_gospa.plot(scans, res_D['gospa'], '--s', color='#CC0000',
                  markersize=5, linewidth=1.5, alpha=0.6, label='MUSIC+Standard')
    ax_gospa.axvspan(7, 11, alpha=0.1, color='red')
    ax_gospa.axvspan(15, 19, alpha=0.1, color='red')
    ax_gospa.set_xlabel('Scan')
    ax_gospa.set_ylabel('GOSPA (degrees)')
    ax_gospa.set_title('(e) GOSPA Metric (lower = better)', fontweight='bold')
    ax_gospa.legend(loc='upper right', fontsize=14)

    # (f) Detection Rate comparison
    ax_pd = fig.add_subplot(gs[2, 1])
    ax_pd.plot(scans, [p*100 for p in res_A['pd']], '-^', color='#0055CC',
               markersize=6, linewidth=2.0, label='COP+Physics (Proposed)')
    ax_pd.plot(scans, [p*100 for p in res_B['pd']], '-s', color='#CC0000',
               markersize=6, linewidth=2.0, label='COP+Standard (Ablation)')
    ax_pd.plot(scans, [p*100 for p in res_C['pd']], '--^', color='#0055CC',
               markersize=5, linewidth=1.5, alpha=0.6, label='MUSIC+Physics')
    ax_pd.plot(scans, [p*100 for p in res_D['pd']], '--s', color='#CC0000',
               markersize=5, linewidth=1.5, alpha=0.6, label='MUSIC+Standard')
    ax_pd.axvspan(7, 11, alpha=0.1, color='red')
    ax_pd.axvspan(15, 19, alpha=0.1, color='red')
    ax_pd.set_xlabel('Scan')
    ax_pd.set_ylabel('Detection Rate (%)')
    ax_pd.set_title('(f) Detection Rate', fontweight='bold')
    ax_pd.set_ylim([0, 110])
    ax_pd.legend(loc='lower right', fontsize=14)

    fig.suptitle(
        'Tracking Ablation: Physics-based vs Standard GM-PHD\n'
        f'M={M}, K={K} crossing targets, SNR={SNR_DB} dB, T={T}',
        fontsize=28, fontweight='bold', y=0.995,
    )

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_tracking_ablation.png'))
    plt.close(fig)
    print(f"\nSaved fig_tracking_ablation.png")


if __name__ == "__main__":
    main()
