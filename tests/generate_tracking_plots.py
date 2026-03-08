#!/usr/bin/env python3
"""Generate tracking benchmark plots for COP-RFS.

Figures:
  Fig 7: COP-RFS tracking - target birth/death
  Fig 8: T-COP + PHD feedback loop comparison
  Fig 9: Moving target tracking
  Fig 10: Proposed algorithm overview diagram
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import SubspaceCOP, TemporalCOP
from iron_dome_sim.tracking import COPPHD, ConstantVelocity
from iron_dome_sim.eval.metrics import rmse_doa, detection_rate

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_birth_death():
    """Fig 7: Target birth and death tracking."""
    print("Generating Fig 7: Birth/Death tracking...")

    M = 8
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 256
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_scans = 25

    cop = SubspaceCOP(array, rho=2, spectrum_type="combined")
    model = ConstantVelocity(dt=0.1, process_noise_std=0.01)
    phd = COPPHD(model, cop, survival_prob=0.90, detection_prob=0.90,
                 birth_weight=0.2, clutter_rate=1.0)

    true_history = []
    est_history = []
    k_true_history = []
    k_est_history = []

    for scan_i in range(n_scans):
        if scan_i < 5:
            K = 3; true_doas = np.radians([-30, 0, 30])
        elif scan_i < 10:
            K = 6; true_doas = np.radians([-50, -30, -10, 10, 30, 50])
        elif scan_i < 15:
            K = 6; true_doas = np.radians([-50, -30, -10, 10, 30, 50])
        elif scan_i < 20:
            K = 4; true_doas = np.radians([-50, -10, 10, 50])
        else:
            K = 2; true_doas = np.radians([-50, 50])

        cop.num_sources = K
        np.random.seed(42 + scan_i)
        X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")
        phd.process_scan(X, scan_angles)
        est_doas = phd.get_doa_estimates()

        true_history.append(true_doas)
        est_history.append(est_doas)
        k_true_history.append(K)
        k_est_history.append(len(est_doas))

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

    # DOA tracks
    scans = np.arange(1, n_scans + 1)
    for i, td in enumerate(true_history):
        for d in td:
            ax1.plot(i + 1, np.degrees(d), 'b+', markersize=8, markeredgewidth=1.5)
    for i, ed in enumerate(est_history):
        for d in ed:
            ax1.plot(i + 1, np.degrees(d), 'ro', markersize=5, alpha=0.7)

    ax1.set_ylabel('DOA (degrees)')
    ax1.set_title('COP-RFS Tracking: Target Birth & Death')
    ax1.legend(['True DOA', 'Estimated DOA'],
               loc='upper right', framealpha=0.9)

    # Annotate phases
    ax1.axvspan(0.5, 5.5, alpha=0.1, color='green')
    ax1.axvspan(5.5, 15.5, alpha=0.1, color='blue')
    ax1.axvspan(15.5, 20.5, alpha=0.1, color='orange')
    ax1.axvspan(20.5, 25.5, alpha=0.1, color='red')
    ax1.text(3, -55, 'K=3', fontsize=9, ha='center', color='green')
    ax1.text(10.5, -55, 'K=6 (birth)', fontsize=9, ha='center', color='blue')
    ax1.text(18, -55, 'K=4 (death)', fontsize=9, ha='center', color='orange')
    ax1.text(23, -55, 'K=2', fontsize=9, ha='center', color='red')

    # Target count
    ax2.step(scans, k_true_history, 'b-', linewidth=2, where='mid', label='True K')
    ax2.step(scans, k_est_history, 'r--', linewidth=2, where='mid', label='Estimated K')
    ax2.set_xlabel('Scan')
    ax2.set_ylabel('Target Count')
    ax2.set_title('Estimated vs True Target Count')
    ax2.legend(loc='upper right')
    ax2.set_ylim([0, 8])

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig7_birth_death.png'))
    plt.close(fig)
    print("  Saved fig7_birth_death.png")


def plot_tcop_phd_feedback():
    """Fig 8: T-COP + PHD feedback loop comparison."""
    print("Generating Fig 8: T-COP + PHD feedback...")

    M = 8
    K = 6
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 5
    T = 128
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_scans = 20
    true_doas = np.radians([-50, -30, -10, 10, 30, 50])

    results = {}
    for label, cop_cls, cop_kwargs in [
        ('COP + PHD', SubspaceCOP, dict(rho=2, num_sources=K, spectrum_type="combined")),
        ('T-COP + PHD', TemporalCOP, dict(rho=2, num_sources=K, alpha=0.85, prior_weight=0.2)),
    ]:
        cop = cop_cls(array, **cop_kwargs)
        model = ConstantVelocity(dt=0.1, process_noise_std=0.01)
        phd = COPPHD(model, cop, survival_prob=0.95, detection_prob=0.90,
                     birth_weight=0.15)

        pds, rmses = [], []
        for scan_i in range(n_scans):
            np.random.seed(42 + scan_i)
            X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")
            phd.process_scan(X, scan_angles)
            est_doas = phd.get_doa_estimates()

            if len(est_doas) > 0:
                rmse_val, _ = rmse_doa(est_doas, true_doas)
                pd, _ = detection_rate(est_doas, true_doas)
            else:
                rmse_val, pd = np.radians(90), 0.0
            pds.append(pd)
            rmses.append(np.degrees(rmse_val))

        results[label] = {'pd': pds, 'rmse': rmses}

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    scans = np.arange(1, n_scans + 1)

    colors = {'COP + PHD': '#1f77b4', 'T-COP + PHD': '#d62728'}
    for label, data in results.items():
        ax1.plot(scans, data['pd'], '-o', color=colors[label], label=label, markersize=4)
        ax2.plot(scans, data['rmse'], '-o', color=colors[label], label=label, markersize=4)

    ax1.set_xlabel('Scan')
    ax1.set_ylabel('Detection Rate (Pd)')
    ax1.set_title('Detection Rate over Scans')
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc='lower right')

    ax2.set_xlabel('Scan')
    ax2.set_ylabel('RMSE (degrees)')
    ax2.set_title('RMSE over Scans')
    ax2.legend(loc='upper right')

    fig.suptitle('T-COP + PHD Feedback Loop (M=8, K=6, SNR=5dB)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig8_tcop_phd_feedback.png'))
    plt.close(fig)
    print("  Saved fig8_tcop_phd_feedback.png")


def plot_algorithm_overview():
    """Fig 10: Proposed algorithm system overview."""
    print("Generating Fig 10: Algorithm overview diagram...")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Proposed Algorithm System: COP-RFS\n'
                 'Underdetermined DOA Estimation + Multi-Target Tracking',
                 fontsize=16, fontweight='bold', pad=20)

    # Helper functions
    def draw_box(x, y, w, h, text, color='lightblue', fontsize=9, bold=False):
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='black',
                              facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, wrap=True)

    def draw_arrow(x1, y1, x2, y2, text='', color='black'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.15, text, ha='center', va='bottom',
                   fontsize=8, color=color)

    # Title boxes for sections
    draw_box(0.2, 7.0, 13.5, 0.7, 'PROPOSED: COP-RFS System', 'gold', 12, True)

    # Row 1: Signal Processing
    draw_box(0.5, 5.5, 2.5, 1.0, 'Radar Signal\nX(t) = A*s(t) + n(t)\nM sensors, K>M sources',
             '#E8F4FD', 8)
    draw_box(3.5, 5.5, 3.0, 1.0, '4th-Order Cumulant\nC_{2rho} (Toeplitz)\nNoise elimination',
             '#FFE4E1', 9, True)
    draw_box(7.0, 5.5, 3.0, 1.0, 'COP Spectrum\nP(theta) = a^H P_s a / a^H P_n a\nUnderdetermined DOA',
             '#FFE4E1', 9, True)
    draw_box(10.5, 5.5, 3.0, 1.0, 'DOA Estimates\nK peaks from spectrum\nK > M resolved!',
             '#E8FFE8', 9)

    draw_arrow(3.0, 6.0, 3.5, 6.0)
    draw_arrow(6.5, 6.0, 7.0, 6.0)
    draw_arrow(10.0, 6.0, 10.5, 6.0)

    # Row 2: Novel Extensions
    draw_box(0.5, 3.5, 3.0, 1.5, 'T-COP (Novel IP)\n'
             'Temporal cumulant\naccumulation\nalpha-weighted\n2-3x RMSE gain',
             '#FFFACD', 8, True)
    draw_box(4.0, 3.5, 3.0, 1.5, 'SD-COP (Novel IP)\n'
             'Sequential deflation\nGlobal refinement\nExtends capacity\nbeyond rho*(M-1)',
             '#FFFACD', 8, True)
    draw_box(7.5, 3.5, 3.0, 1.5, 'COP-RFS (Novel IP)\n'
             'GM-PHD filter\nCOP spectrum birth\nNo data association\nRFS framework',
             '#FFFACD', 8, True)
    draw_box(11.0, 3.5, 2.5, 1.5, 'Iron Dome\nSimulation\nMulti-radar\n50+ threats\nInterception',
             '#E8FFE8', 8)

    # Arrows from Row 1 to Row 2
    draw_arrow(2.0, 5.5, 2.0, 5.0, 'temporal', '#d62728')
    draw_arrow(5.5, 5.5, 5.5, 5.0, 'deflation', '#2ca02c')
    draw_arrow(9.0, 5.5, 9.0, 5.0, 'birth intensity', '#9467bd')

    # Row 3: Feedback loop
    draw_box(3.5, 1.5, 4.5, 1.5, 'Tracker Feedback Loop\n\n'
             'PHD estimates -> T-COP prior\n'
             'Predicted DOAs constrain subspace\n'
             'Adaptive signal dimension',
             '#FFD1DC', 9, True)

    # Feedback arrows
    draw_arrow(9.0, 3.5, 8.0, 3.0, 'track\nestimates', '#9467bd')
    draw_arrow(5.75, 3.0, 5.75, 3.5, '', '#9467bd')
    draw_arrow(3.5, 2.25, 2.0, 3.5, 'constrain\nsubspace', '#d62728')

    # Arrow from COP-RFS to Iron Dome
    draw_arrow(10.5, 4.25, 11.0, 4.25)

    # Patent claims
    draw_box(9.0, 1.5, 4.5, 1.5, 'Patent Claims\n'
             '1. Temporal cumulant accumulation\n'
             '2. Sequential deflation in HOC\n'
             '3. COP spectrum as PHD birth\n'
             '4. Tracker-aided subspace constraint',
             '#F0E68C', 8)

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig10_algorithm_overview.png'))
    plt.close(fig)
    print("  Saved fig10_algorithm_overview.png")


def plot_moving_targets():
    """Fig 9: Moving target tracking with COP-RFS."""
    print("Generating Fig 9: Moving target tracking...")

    M = 8
    K = 4
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 256
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_scans = 25
    dt = 0.1

    tcop = TemporalCOP(array, rho=2, num_sources=K, alpha=0.85,
                        prior_weight=0.2)
    model = ConstantVelocity(dt=dt, process_noise_std=0.01)
    phd = COPPHD(model, tcop, survival_prob=0.95, detection_prob=0.90,
                 birth_weight=0.15, clutter_rate=1.0)

    base_doas = np.radians([-50, -20, 10, 40])
    rates = np.radians([2.0, -1.5, 1.0, -2.5])

    true_tracks = []
    est_tracks = []

    for scan_i in range(n_scans):
        true_doas = base_doas + rates * scan_i
        true_doas = np.clip(true_doas, -np.pi/2 + 0.05, np.pi/2 - 0.05)

        np.random.seed(42 + scan_i)
        X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")
        phd.process_scan(X, scan_angles)
        est_doas = phd.get_doa_estimates()

        true_tracks.append(true_doas)
        est_tracks.append(est_doas)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scans = np.arange(1, n_scans + 1)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for k in range(K):
        true_doas_k = [np.degrees(true_tracks[i][k]) for i in range(n_scans)]
        ax.plot(scans, true_doas_k, '-', color=colors[k], linewidth=2,
                label=f'True target {k+1}' if k == 0 else None)

    # Plot all estimated DOAs
    for i in range(n_scans):
        for d in est_tracks[i]:
            ax.plot(i + 1, np.degrees(d), 'rx', markersize=6, alpha=0.7)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2, label='True tracks'),
        Line2D([0], [0], marker='x', color='red', linewidth=0,
               markersize=6, label='T-COP + PHD estimates'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlabel('Scan')
    ax.set_ylabel('DOA (degrees)')
    ax.set_title('T-COP + PHD Moving Target Tracking\n'
                 'M=8, K=4 targets, SNR=15dB')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig9_moving_targets.png'))
    plt.close(fig)
    print("  Saved fig9_moving_targets.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Tracking Benchmark Plots")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    plot_birth_death()
    plot_tcop_phd_feedback()
    plot_moving_targets()
    plot_algorithm_overview()

    print(f"\nAll tracking plots saved to: {OUTPUT_DIR}")
    print("Done!")
