#!/usr/bin/env python3
"""Generate rho scaling plot: COP performance vs cumulant order.

Shows the crossover pattern:
  rho=1 (2nd order) best for K <= M-1 = 7
  rho=2 (4th order) best for K = 8-14
  rho=3 (6th order) best for K >= 15
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
from iron_dome_sim.doa import SubspaceCOP

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 22,
    'axes.labelsize': 25,
    'axes.titlesize': 25,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
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
    snr_db = 20
    T = 1024
    seeds = [42, 55, 77]

    rho_values = [1, 2, 3]
    K_values = [3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]

    # Results: detection rate per (rho, K)
    results = {rho: [] for rho in rho_values}

    for K in K_values:
        true_doas = np.radians(np.linspace(-65, 65, K))

        for rho in rho_values:
            capacity = rho * (M - 1)
            num_src = min(K, capacity)

            dets = []
            for seed in seeds:
                np.random.seed(seed)
                X, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                             "non_stationary")
                cop = SubspaceCOP(array, rho=rho, num_sources=num_src,
                                  spectrum_type="combined")
                est, _ = cop.estimate(X, scan_angles)
                dets.append(count_correct(est, true_doas) / K)

            results[rho].append(np.mean(dets))

        print(f"  K={K:2d}: " + "  ".join(
            f"rho={rho}: {results[rho][-1]*100:5.1f}%" for rho in rho_values))

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {'1': '#1f77b4', '2': '#d62728', '3': '#2ca02c'}
    markers = {'1': 'o', '2': 's', '3': 'D'}
    labels = {
        1: r'$\rho=1$ (2nd order, $K_{\max}=7$)',
        2: r'$\rho=2$ (4th order, $K_{\max}=14$)',
        3: r'$\rho=3$ (6th order, $K_{\max}=21$)',
    }

    for rho in rho_values:
        ax.plot(K_values, [r * 100 for r in results[rho]],
                '-' + markers[str(rho)], color=colors[str(rho)],
                markersize=10, markeredgecolor='black', markeredgewidth=0.8,
                label=labels[rho], linewidth=2.5)

    # Capacity lines
    for rho in rho_values:
        cap = rho * (M - 1)
        ax.axvline(x=cap, color=colors[str(rho)], linestyle='--',
                   alpha=0.4, linewidth=1.5)
        ax.text(cap + 0.3, 105, f'$K_{{max}}^{{\\rho={rho}}}={cap}$',
                fontsize=15, color=colors[str(rho)], va='bottom')

    ax.set_xlabel('Number of Sources (K)')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title(f'COP Detection Rate vs Cumulant Order $\\rho$\n'
                 f'[M={M}, SNR={snr_db} dB, T={T}]',
                 fontweight='bold')
    ax.set_ylim([-5, 115])
    ax.set_xlim([2, 21])
    ax.legend(loc='upper right', fontsize=20)

    # Annotate regions
    ax.axvspan(2, 7, alpha=0.05, color='blue')
    ax.axvspan(7, 14, alpha=0.05, color='red')
    ax.axvspan(14, 21, alpha=0.05, color='green')
    ax.text(4.5, -12, r'$\rho=1$ best', fontsize=17, ha='center',
            color='#1f77b4', fontweight='bold')
    ax.text(10.5, -12, r'$\rho=2$ best', fontsize=17, ha='center',
            color='#d62728', fontweight='bold')
    ax.text(17.5, -12, r'$\rho=3$ best', fontsize=17, ha='center',
            color='#2ca02c', fontweight='bold')

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_rho_scaling.png'))
    plt.close(fig)
    print(f"\nSaved fig_rho_scaling.png")


if __name__ == '__main__':
    print("Generating rho scaling plot...")
    main()
    print("Done!")
