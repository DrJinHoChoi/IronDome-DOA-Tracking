#!/usr/bin/env python3
"""Generate signal type comparison plot for paper.

Shows COP advantage over MUSIC for different non-stationary signal types.
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
from iron_dome_sim.doa import SubspaceCOP, MUSIC

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 14,
    'lines.linewidth': 2.5,
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


def count_correct(est, true, thr_deg=3.0):
    if len(est) == 0:
        return 0
    return sum(1 for d in true if min(abs(est - d)) < np.radians(thr_deg))


def main():
    M = 8
    array = UniformLinearArray(M=M, d=0.5)
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 3601)
    K = 10
    snr_db = 15
    T = 512
    seeds = [42, 55, 77, 123, 300]
    true_doas = np.radians(np.linspace(-54, 54, K))

    signal_types = ["stationary", "non_stationary", "speech", "fm",
                    "chirp", "psk", "missile", "mixed"]
    display_names = ["Gaussian\n(i.i.d.)", "AM\nSinusoid", "Speech\n(Voiced)", "FM\nModulation",
                     "Chirp\n(LFM)", "QPSK\nComm.", "Missile\n(Swerling)", "Mixed"]

    cop_results = []
    music_results = []

    for sig_type in signal_types:
        cop_dets = []
        music_dets = []
        for seed in seeds:
            np.random.seed(seed)
            X, _, _ = generate_snapshots(array, true_doas, snr_db, T, sig_type)

            cop = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
            cop_est, _ = cop.estimate(X, scan_angles)
            cop_dets.append(count_correct(cop_est, true_doas) / K * 100)

            music = MUSIC(array, num_sources=min(K, M - 1))
            music_est, _ = music.estimate(X, scan_angles)
            music_dets.append(count_correct(music_est, true_doas) / K * 100)

        cop_results.append(np.mean(cop_dets))
        music_results.append(np.mean(music_dets))

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(signal_types))
    width = 0.35

    bars_cop = ax.bar(x - width/2, cop_results, width, label='COP-4th (Proposed)',
                       color='#0055CC', edgecolor='black', linewidth=0.8, zorder=3)
    bars_music = ax.bar(x + width/2, music_results, width, label='MUSIC',
                         color='#AAAAAA', edgecolor='black', linewidth=0.8, zorder=3)

    # Annotate bars with values
    for bar, val in zip(bars_cop, cop_results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=11,
                fontweight='bold', color='#0055CC')
    for bar, val in zip(bars_music, music_results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10,
                color='#666666')

    ax.set_xlabel('Signal Type')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title(f'COP vs MUSIC: Detection Rate by Signal Type\n'
                 f'[M={M}, K={K} (underdetermined), SNR={snr_db} dB, T={T}]',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylim([0, 115])
    ax.legend(loc='upper right', fontsize=14)

    # Highlight non-Gaussian advantage
    ax.axhline(y=100 * (M-1)/K, color='red', linestyle='--', alpha=0.5,
               linewidth=1.5, label=f'MUSIC capacity limit ({M-1}/{K})')

    # Annotate key findings
    ax.annotate('Non-Gaussian signals:\nCOP resolves K > M-1',
                xy=(3, 95), fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_signal_types.png'))
    plt.close(fig)
    print("Saved fig_signal_types.png")


if __name__ == '__main__':
    main()
