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
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 2.5,
    'lines.markersize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
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
    model = ConstantVelocity(dt=1.0, process_noise_std=np.radians(0.5))
    phd = COPPHD(model, cop,
                 survival_prob=0.95,
                 detection_prob=0.95,
                 birth_weight=1.0,    # Instant confirmation at SNR=15dB
                 clutter_rate=0.3,
                 prune_threshold=1e-3,
                 merge_threshold=2.0,
                 birth_pos_std_deg=2.0,
                 birth_vel_std_deg=3.0,
                 association_gate_deg=8.0)

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
    from matplotlib.lines import Line2D
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])

    # DOA tracks
    scans = np.arange(1, n_scans + 1)
    # True DOAs: GREEN filled circles (consistent with spectrum plots)
    for i, td in enumerate(true_history):
        for d in td:
            ax1.plot(i + 1, np.degrees(d), 'o', color='#00AA00', markersize=11,
                     markeredgecolor='darkgreen', markeredgewidth=1.2, zorder=4)
    # Estimated DOAs: RED inverted triangles (consistent with spectrum plots)
    for i, ed in enumerate(est_history):
        for d in ed:
            ax1.plot(i + 1, np.degrees(d), 'v', color='#DD0000', markersize=9,
                     alpha=0.8, markeredgecolor='black', markeredgewidth=0.8, zorder=5)

    ax1.set_ylabel('DOA (degrees)')
    ax1.set_title('COP-RFS Tracking: Target Birth & Death', fontweight='bold')
    legend_bd = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00AA00',
               markeredgecolor='darkgreen', markersize=12, markeredgewidth=1.2,
               label='True DOA'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#DD0000',
               markeredgecolor='black', markersize=10, markeredgewidth=0.8,
               label='Estimated DOA'),
    ]
    ax1.legend(handles=legend_bd, loc='upper right', framealpha=0.9)

    # Annotate phases
    ax1.axvspan(0.5, 5.5, alpha=0.08, color='green')
    ax1.axvspan(5.5, 15.5, alpha=0.08, color='blue')
    ax1.axvspan(15.5, 20.5, alpha=0.08, color='orange')
    ax1.axvspan(20.5, 25.5, alpha=0.08, color='red')
    ax1.text(3, -55, 'K=3', fontsize=15, ha='center', color='green', fontweight='bold')
    ax1.text(10.5, -55, 'K=6 (birth)', fontsize=15, ha='center', color='blue', fontweight='bold')
    ax1.text(18, -55, 'K=4 (death)', fontsize=15, ha='center', color='orange', fontweight='bold')
    ax1.text(23, -55, 'K=2', fontsize=15, ha='center', color='red', fontweight='bold')

    # Target count
    ax2.step(scans, k_true_history, 'b-', linewidth=2.5, where='mid', label='True K')
    ax2.step(scans, k_est_history, 'r--', linewidth=2.5, where='mid', label='Estimated K')
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
        model = ConstantVelocity(dt=1.0, process_noise_std=np.radians(0.5))
        phd = COPPHD(model, cop,
                     survival_prob=0.98,
                     detection_prob=0.95,
                     birth_weight=1.0,    # Faster confirmation
                     clutter_rate=0.3,
                     prune_threshold=1e-3,
                     merge_threshold=2.0,
                     birth_pos_std_deg=2.0,
                     birth_vel_std_deg=3.0,
                     association_gate_deg=8.0)

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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    scans = np.arange(1, n_scans + 1)

    colors = {'COP + PHD': '#1f77b4', 'T-COP + PHD': '#d62728'}
    for label, data in results.items():
        ax1.plot(scans, data['pd'], '-o', color=colors[label], label=label, markersize=8, linewidth=2.5)
        ax2.plot(scans, data['rmse'], '-o', color=colors[label], label=label, markersize=8, linewidth=2.5)

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
                 fontsize=22, fontweight='bold', y=1.02)
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
    """Fig 9: Moving target tracking with COP-RFS.

    Key parameter choices:
    - dt=1.0: Each scan is one time step (scan-by-scan tracking)
    - process_noise_std=1.5deg: Covers target maneuver range
    - birth_vel_std_deg=5: Initial velocity uncertainty covers +-5 deg/scan
    - survival_prob=0.99: High persistence across scans
    - Velocity-gated merge: prevents track loss at crossings

    Target crossings occur at:
    - Scan ~9: Target 1 & 2 cross (-33deg), Target 3 & 4 cross (19deg)
    - Scan ~20: Target 1 & 4 cross (-10deg)
    The velocity-gated merge in PHD preserves both tracks through crossings.
    """
    print("Generating Fig 9: Moving target tracking...")

    M = 8
    K = 4
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 512  # More snapshots for better COP resolution at crossings
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_scans = 30
    dt = 1.0  # One scan = one time step

    # Use standard COP (NOT T-COP) for moving targets.
    # T-COP's temporal accumulation smears the spectrum for moving targets.
    cop = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    # Low process noise (0.3 deg/scan): once velocity is learned (scans 3-5),
    # the filter trusts its prediction through crossings where COP
    # measurements become unreliable. High birth_vel_std allows initial
    # velocity learning despite low process noise.
    model = ConstantVelocity(dt=dt, process_noise_std=np.radians(0.3))
    phd = COPPHD(model, cop,
                 survival_prob=0.99,
                 detection_prob=0.95,
                 birth_weight=0.3,
                 clutter_rate=0.2,
                 prune_threshold=1e-3,
                 merge_threshold=2.0,
                 birth_pos_std_deg=2.0,
                 birth_vel_std_deg=8.0)

    # Target scenario: 4 crossing targets
    # Crossings at scan ~9 (T1-T2, T3-T4) and scan ~20 (T1-T4)
    base_doas = np.radians([-50, -20, 10, 40])
    rates = np.radians([2.0, -1.5, 1.0, -2.5])  # deg/scan

    true_tracks = []
    est_tracks = []
    track_label_history = []  # Per-scan: dict {label: (state, cov, weight)}
    rmse_per_scan = []

    for scan_i in range(n_scans):
        true_doas = base_doas + rates * scan_i
        true_doas = np.clip(true_doas, -np.pi/2 + 0.05, np.pi/2 - 0.05)

        np.random.seed(42 + scan_i)
        X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")
        phd.process_scan(X, scan_angles)
        est_doas = phd.get_doa_estimates()

        # Get per-track states with labels for identification
        track_states = phd.get_track_states()
        track_label_history.append(track_states)

        true_tracks.append(true_doas)
        est_tracks.append(est_doas)

        if len(est_doas) > 0:
            r, _ = rmse_doa(est_doas, true_doas)
            rmse_per_scan.append(np.degrees(r))
        else:
            rmse_per_scan.append(90.0)

    # === Track-to-source identification ===
    # Use BOTH position AND velocity for robust assignment through crossings.
    # Velocity is the key discriminant when targets cross (same position,
    # different velocities).
    from scipy.optimize import linear_sum_assignment

    all_labels = set()
    for tlh in track_label_history:
        all_labels.update(tlh.keys())
    all_labels = sorted(all_labels)

    # For each label, find best matching source using position + velocity cost
    label_to_source = {}
    source_assigned = {}  # Track which source has best-matching label

    for label in all_labels:
        # Collect all states for this label across scans
        states_for_label = []
        scan_indices = []
        for scan_i, tlh in enumerate(track_label_history):
            if label in tlh:
                states_for_label.append(tlh[label][0])  # full state [az, el, vaz, vel]
                scan_indices.append(scan_i)
        if len(states_for_label) == 0:
            continue

        # Find best matching source using position + velocity
        best_source = -1
        best_cost = float('inf')
        for k in range(K):
            cost = 0.0
            for state, si in zip(states_for_label, scan_indices):
                true_doa = base_doas[k] + rates[k] * si
                true_doa = np.clip(true_doa, -np.pi/2 + 0.05, np.pi/2 - 0.05)
                pos_err = abs(state[0] - true_doa)
                # Velocity cost: compare estimated velocity with true rate
                vel_err = abs(state[2] - rates[k]) if len(state) > 2 else 0
                # Combined cost: position + velocity (velocity weighted higher
                # to resolve crossings where positions are identical)
                cost += pos_err + 2.0 * vel_err
            avg_cost = cost / len(states_for_label)
            if avg_cost < best_cost:
                best_cost = avg_cost
                best_source = k
        label_to_source[label] = best_source

    # Resolve conflicts: if multiple labels map to same source, keep best
    source_best_label = {}
    source_best_cost = {}
    for label, source in label_to_source.items():
        # Compute total scan count for this label
        n_scans_label = sum(1 for tlh in track_label_history if label in tlh)
        if source not in source_best_label or n_scans_label > source_best_cost.get(source, 0):
            source_best_label[source] = label
            source_best_cost[source] = n_scans_label

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])
    scans = np.arange(1, n_scans + 1)

    # Color per target
    true_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    est_markers = ['o', 's', 'D', '^']

    # TRUE TRACKS: SOLID THICK LINES with alpha (clearly "ground truth")
    for k in range(K):
        true_doas_k = [np.degrees(true_tracks[i][k]) for i in range(n_scans)]
        ax1.plot(scans, true_doas_k, '-', color=true_colors[k], linewidth=3.5,
                 alpha=0.5, zorder=2)

    # Mark crossing points
    for cross_scan in [9, 20]:
        if cross_scan < n_scans:
            ax1.axvline(x=cross_scan, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)

    # ESTIMATED DOAs: MARKERS with BLACK EDGE (clearly "estimated")
    n_plotted = 0
    n_unassigned = 0
    for scan_i, tlh in enumerate(track_label_history):
        for label, (state, cov, w) in tlh.items():
            doa_deg = np.degrees(state[0])
            source_id = label_to_source.get(label, -1)
            if source_id >= 0 and source_id < K:
                c = true_colors[source_id]
                m = est_markers[source_id]
                n_plotted += 1
            else:
                c = 'gray'
                m = 'x'
                n_unassigned += 1
            ax1.plot(scan_i + 1, doa_deg, m, color=c, markersize=12,
                    markeredgecolor='black', markeredgewidth=1.2, alpha=0.9, zorder=5)

    # Legend: CLEARLY separate True (lines) from Estimated (markers)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=4.0, alpha=0.5,
               linestyle='-', label='True Track (line)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=11, markeredgewidth=1.2,
               label='Estimated DOA (marker)'),
    ]
    for k in range(K):
        legend_elements.append(
            Line2D([0], [0], color=true_colors[k], linewidth=3.0,
                   marker=est_markers[k], markersize=10,
                   markeredgecolor='black', markeredgewidth=0.8,
                   label=f'Target {k+1}'))
    if n_unassigned > 0:
        legend_elements.append(
            Line2D([0], [0], marker='x', color='gray', linewidth=0,
                   markersize=10, label='Unassigned'))
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=12)
    ax1.set_ylabel('DOA (degrees)')
    ax1.set_title('COP-RFS Moving Target Tracking: Physics-Based Identification\n'
                  f'M={M}, K={K} targets, SNR={snr_db}dB, {n_scans} scans '
                  '(predict \u2192 identify \u2192 update)',
                  fontweight='bold')
    ax1.set_xlabel('Scan')

    # Annotate crossings
    ax1.annotate('T1-T2\ncross', xy=(9, -33), fontsize=13, ha='center',
                color='gray', alpha=0.7, fontweight='bold')
    ax1.annotate('T1-T4\ncross', xy=(20, -10), fontsize=13, ha='center',
                color='gray', alpha=0.7, fontweight='bold')

    # RMSE subplot
    ax2.plot(scans, rmse_per_scan, 'b-o', markersize=6, linewidth=2.0)
    ax2.set_xlabel('Scan')
    ax2.set_ylabel('RMSE (deg)')
    ax2.set_title('Tracking RMSE per Scan')
    rmse_after_warmup = rmse_per_scan[2:] if len(rmse_per_scan) > 2 else rmse_per_scan
    y_max = max(max(rmse_after_warmup) * 1.5, 3.0)
    ax2.set_ylim([0, y_max])
    avg_rmse = np.mean(rmse_per_scan[3:])  # Skip warmup
    ax2.axhline(y=avg_rmse, color='red', linestyle='--', alpha=0.7, linewidth=2.0,
                label=f'Avg RMSE (after warmup): {avg_rmse:.2f} deg')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig9_moving_targets.png'))
    plt.close(fig)
    print(f"  Saved fig9_moving_targets.png (avg RMSE={avg_rmse:.2f} deg)")
    print(f"    Plotted: {n_plotted} assigned, {n_unassigned} unassigned")


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
