#!/usr/bin/env python3
"""End-to-end comparison: COP-RFS vs MUSIC-PHD pipeline.

KEY experiment demonstrating the unified COP-RFS framework advantage.

Scenario: K=10 moving targets (underdetermined: K > M-1 = 7).
  MUSIC is fundamentally limited to resolving M-1=7 sources.
  COP with rho=2 resolves up to rho*(M-1)=14 sources.

Metrics: Detection Rate (Pd), RMSE, and GOSPA (penalizes missed targets).

Output: results/figures/fig_end2end_comparison.png
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
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
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
})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Parameters ──────────────────────────────────────────────────────────
M = 8
K = 10  # Underdetermined: K > M-1 = 7
SNR_DB = 15
T = 1024
N_SCANS = 30
DT = 1.0
SCAN_ANGLES = np.linspace(-np.pi / 2, np.pi / 2, 1801)

# K=10 moving targets: widely spaced with varying velocities
BASE_DOAS = np.radians(np.linspace(-72, 72, K))
RATES = np.radians([1.2, -0.8, 0.6, -1.0, 1.5, -1.3, 0.9, -0.7, 1.1, -1.4])

TARGET_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f',
]
TARGET_MARKERS = ['o', 's', 'D', '^', 'P', 'X', 'h', '*', 'v', '<']


def build_phd(estimator):
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
    )


def run_pipeline(label, estimator):
    """Run full tracking pipeline and collect per-scan metrics."""
    array = UniformLinearArray(M=M, d=0.5)
    phd = build_phd(estimator)

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

        # GOSPA (penalizes missed AND false targets)
        g, _ = gospa(
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
                + 2.0 * (abs(s[2] - rates[k]) if len(s)>2 else 0)
                for s, si in zip(states, sids)
            ) / len(states)
            if c < best_c:
                best_c, best_k = c, k
        l2s[label] = best_k
    return l2s


def plot_tracking(ax, true_all, label_hist, l2s, K, title):
    """Plot DOA tracking panel."""
    scans = np.arange(1, N_SCANS + 1)

    # True tracks: solid translucent lines
    for k in range(K):
        doas = [np.degrees(true_all[i][k]) for i in range(N_SCANS)]
        ax.plot(scans, doas, '-', color=TARGET_COLORS[k % len(TARGET_COLORS)],
                linewidth=3.0, alpha=0.4, zorder=2)

    # Estimated: markers
    for si, tlh in enumerate(label_hist):
        for label, (state, _, _) in tlh.items():
            src = l2s.get(label, -1)
            if 0 <= src < K:
                c = TARGET_COLORS[src % len(TARGET_COLORS)]
                m = TARGET_MARKERS[src % len(TARGET_MARKERS)]
            else:
                c, m = 'gray', 'x'
            ax.plot(si + 1, np.degrees(state[0]), m, color=c, markersize=9,
                    markeredgecolor='black', markeredgewidth=0.8,
                    alpha=0.85, zorder=5)

    ax.set_ylabel('DOA (degrees)')
    ax.set_xlabel('Scan')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim([0.5, N_SCANS + 0.5])


def main():
    print("=" * 70)
    print("End-to-End: COP-RFS vs MUSIC-PHD")
    print(f"  M={M}, K={K} (underdetermined: K > M-1={M-1})")
    print(f"  SNR={SNR_DB} dB, T={T}, {N_SCANS} scans")
    print("=" * 70)

    array = UniformLinearArray(M=M, d=0.5)

    # Pipeline 1: COP-RFS
    print("\nRunning COP-RFS pipeline...")
    cop = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    res_cop = run_pipeline("COP-RFS", cop)

    # Pipeline 2: MUSIC-PHD
    print("Running MUSIC-PHD pipeline...")
    music = MUSIC(array, num_sources=min(K, M - 1))  # capped at M-1=7
    res_mus = run_pipeline("MUSIC-PHD", music)

    # Label assignment
    l2s_cop = assign_labels(res_cop['labels'], K, BASE_DOAS, RATES)
    l2s_mus = assign_labels(res_mus['labels'], K, BASE_DOAS, RATES)

    # ── Summary ─────────────────────────────────────────────────────
    avg_pd_cop = np.mean(res_cop['pd'][3:])
    avg_pd_mus = np.mean(res_mus['pd'][3:])
    avg_rmse_cop = np.mean(res_cop['rmse'][3:])
    avg_rmse_mus = np.mean(res_mus['rmse'][3:])
    avg_gospa_cop = np.mean(res_cop['gospa'][3:])
    avg_gospa_mus = np.mean(res_mus['gospa'][3:])

    print(f"\n{'Metric':<20s} {'COP-RFS':>10s} {'MUSIC-PHD':>10s}")
    print("-" * 45)
    print(f"{'Avg Pd (%)':.<20s} {avg_pd_cop*100:>9.1f}% {avg_pd_mus*100:>9.1f}%")
    print(f"{'Avg RMSE (deg)':.<20s} {avg_rmse_cop:>10.2f} {avg_rmse_mus:>10.2f}")
    print(f"{'Avg GOSPA (deg)':.<20s} {avg_gospa_cop:>10.2f} {avg_gospa_mus:>10.2f}")

    # ── Figure: 2x2 layout ─────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25,
                          height_ratios=[3, 2])

    # (a) COP-RFS tracking
    ax_cop = fig.add_subplot(gs[0, 0])
    plot_tracking(ax_cop, res_cop['true'], res_cop['labels'], l2s_cop, K,
                  f'(a) COP-RFS: {K} targets tracked (K > M$-$1={M-1})')

    leg_cop = [
        Line2D([0], [0], color='gray', linewidth=3.5, alpha=0.5,
               label='True track'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=9, label='Estimated DOA'),
    ]
    ax_cop.legend(handles=leg_cop, loc='upper right', fontsize=11)

    # (b) MUSIC-PHD tracking
    ax_mus = fig.add_subplot(gs[0, 1])
    plot_tracking(ax_mus, res_mus['true'], res_mus['labels'], l2s_mus, K,
                  f'(b) MUSIC-PHD: {K} targets (limit={M-1})')

    leg_mus = [
        Line2D([0], [0], color='gray', linewidth=3.5, alpha=0.5,
               label='True track'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=9, label='Estimated DOA'),
    ]
    ax_mus.legend(handles=leg_mus, loc='upper right', fontsize=11)

    # Annotation: MUSIC limitation
    ax_mus.text(0.5, 0.92, f'MUSIC: max {M-1} of {K} sources resolvable',
                transform=ax_mus.transAxes, fontsize=13, ha='center',
                color='#CC0000', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#FFEEEE', alpha=0.7))

    # (c) Detection Rate
    scans = np.arange(1, N_SCANS + 1)
    ax_pd = fig.add_subplot(gs[1, 0])
    ax_pd.plot(scans, [p*100 for p in res_cop['pd']], '-^', color='#0055CC',
               markersize=6, linewidth=2.0, label='COP-RFS')
    ax_pd.plot(scans, [p*100 for p in res_mus['pd']], '-s', color='#CC0000',
               markersize=6, linewidth=2.0, label='MUSIC-PHD')

    ax_pd.axhline(y=avg_pd_cop*100, color='#0055CC', linestyle='--', alpha=0.5)
    ax_pd.axhline(y=avg_pd_mus*100, color='#CC0000', linestyle='--', alpha=0.5)
    ax_pd.axhline(y=(M-1)/K*100, color='gray', linestyle=':', alpha=0.5,
                  linewidth=1.5, label=f'MUSIC capacity ({M-1}/{K}={100*(M-1)/K:.0f}%)')

    ax_pd.set_xlabel('Scan')
    ax_pd.set_ylabel('Detection Rate (%)')
    ax_pd.set_title(f'(c) Detection Rate: COP-RFS={avg_pd_cop*100:.0f}% vs '
                    f'MUSIC-PHD={avg_pd_mus*100:.0f}%', fontweight='bold')
    ax_pd.set_ylim([0, 110])
    ax_pd.legend(loc='lower right', fontsize=11)

    # (d) GOSPA
    ax_gospa = fig.add_subplot(gs[1, 1])
    ax_gospa.plot(scans, res_cop['gospa'], '-^', color='#0055CC',
                  markersize=6, linewidth=2.0, label='COP-RFS')
    ax_gospa.plot(scans, res_mus['gospa'], '-s', color='#CC0000',
                  markersize=6, linewidth=2.0, label='MUSIC-PHD')

    ax_gospa.axhline(y=avg_gospa_cop, color='#0055CC', linestyle='--', alpha=0.5)
    ax_gospa.axhline(y=avg_gospa_mus, color='#CC0000', linestyle='--', alpha=0.5)

    ax_gospa.text(N_SCANS + 0.5, avg_gospa_cop, f'{avg_gospa_cop:.1f}',
                  color='#0055CC', fontsize=11, va='center')
    ax_gospa.text(N_SCANS + 0.5, avg_gospa_mus, f'{avg_gospa_mus:.1f}',
                  color='#CC0000', fontsize=11, va='center')

    ax_gospa.set_xlabel('Scan')
    ax_gospa.set_ylabel('GOSPA (degrees)')
    ax_gospa.set_title(f'(d) GOSPA: COP-RFS={avg_gospa_cop:.1f}° '
                       f'vs MUSIC-PHD={avg_gospa_mus:.1f}°', fontweight='bold')
    ax_gospa.legend(loc='upper right', fontsize=11)

    fig.suptitle(
        'End-to-End Pipeline: COP-RFS vs MUSIC-PHD\n'
        f'M={M}, K={K} (underdetermined: K > M$-$1={M-1}), '
        f'SNR={SNR_DB} dB, T={T}',
        fontsize=20, fontweight='bold', y=0.995,
    )

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_end2end_comparison.png'))
    plt.close(fig)
    print(f"\nSaved fig_end2end_comparison.png")


if __name__ == "__main__":
    main()
