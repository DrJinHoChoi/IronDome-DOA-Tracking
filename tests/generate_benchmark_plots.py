#!/usr/bin/env python3
"""Generate publication-quality benchmark plots for COP family algorithms.

Generates 6 figures:
  Fig 1: K Scaling - Pd vs K (underdetermined capability)
  Fig 2: K Scaling - RMSE vs K
  Fig 3: SNR Robustness - Pd vs SNR (T-COP temporal accumulation)
  Fig 4: SNR Robustness - RMSE vs SNR
  Fig 5: Close-Spacing Resolution - Pd vs spacing
  Fig 6: Snapshot Efficiency - RMSE vs T
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
from iron_dome_sim.doa import (SubspaceCOP, TemporalCOP, SequentialDeflationCOP,
                                MUSIC, ESPRIT, Capon)
from iron_dome_sim.eval.metrics import rmse_doa, detection_rate

# ============================================================
# Style configuration
# ============================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Algorithm styles: (color, marker, linestyle, label)
# M=8 sensors -> conventional max K=M-1=7, COP max K=rho*(M-1)=14
ALG_STYLES = {
    'MUSIC':     ('#7f7f7f', 's', '--', 'MUSIC (max K=M-1=7)'),
    'ESPRIT':    ('#bcbd22', 'D', '--', 'ESPRIT (max K=M-1=7)'),
    'Capon':     ('#17becf', '^', '--', 'Capon (max K=M-1=7)'),
    'COP':       ('#1f77b4', 'o', '-',  'COP-4th (max K=14) [Proposed]'),
    'T-COP(1)':  ('#ff7f0e', 'v', ':',  'T-COP-4th, 1 scan'),
    'T-COP(5)':  ('#d62728', 'P', '-',  'T-COP-4th, 5 scans [Proposed]'),
    'T-COP(10)': ('#9467bd', '*', '-',  'T-COP-4th, 10 scans [Proposed]'),
    'SD-COP':    ('#2ca02c', 'X', '-.',  'SD-COP-4th (max K>14) [Proposed]'),
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_trial(alg, X, scan_angles, true_doas, n_scans=1, snr_db=15):
    """Run single trial. For T-COP, simulates multi-scan accumulation."""
    try:
        if isinstance(alg, TemporalCOP) and n_scans > 1:
            M, T = X.shape
            for scan_i in range(n_scans - 1):
                np.random.seed(1000 + scan_i)
                X_scan, _, _ = generate_snapshots(
                    alg.array, true_doas, snr_db, T, "non_stationary")
                alg.estimate(X_scan, scan_angles)
        doa_est, _ = alg.estimate(X, scan_angles)
        rmse_val, _ = rmse_doa(doa_est, true_doas)
        pd, _ = detection_rate(doa_est, true_doas)
        return pd, np.degrees(rmse_val)
    except Exception:
        return 0.0, 90.0


def make_alg(name, array, K, snr_db=15):
    """Create algorithm instance by name."""
    M = array.M
    if name == 'MUSIC':
        return MUSIC(array, num_sources=min(K, M - 1))
    elif name == 'ESPRIT':
        return ESPRIT(array, num_sources=min(K, M - 1))
    elif name == 'Capon':
        return Capon(array, num_sources=min(K, M - 1))
    elif name == 'COP':
        return SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    elif name.startswith('T-COP'):
        alg = TemporalCOP(array, rho=2, num_sources=K, alpha=0.85)
        return alg
    elif name == 'SD-COP':
        return SequentialDeflationCOP(array, rho=2, num_sources=K)
    raise ValueError(f"Unknown algorithm: {name}")


def get_n_scans(name):
    """Get number of scans for T-COP variants."""
    if name.startswith('T-COP('):
        return int(name.split('(')[1].rstrip(')'))
    return 1


# ============================================================
# Data Collection Functions
# ============================================================

def collect_k_scaling():
    """Collect K scaling data."""
    print("Collecting K scaling data...")
    M = 8
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 256
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_trials = 10

    K_values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    alg_names = ['MUSIC', 'ESPRIT', 'Capon', 'COP', 'T-COP(5)', 'SD-COP']

    results = {name: {'pd': [], 'rmse': []} for name in alg_names}

    for K in K_values:
        true_doas = np.radians(np.linspace(-55, 55, K))
        print(f"  K={K}...", end=' ', flush=True)

        for name in alg_names:
            pds, rmses = [], []
            for trial in range(n_trials):
                np.random.seed(trial * 100 + K)
                X, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                             "non_stationary")
                alg = make_alg(name, array, K, snr_db)
                pd, rmse = run_trial(alg, X, scan_angles, true_doas,
                                     n_scans=get_n_scans(name), snr_db=snr_db)
                pds.append(pd)
                rmses.append(rmse)

            results[name]['pd'].append(np.mean(pds))
            results[name]['rmse'].append(np.mean(rmses))

        print("done")

    return K_values, results


def collect_snr():
    """Collect SNR robustness data."""
    print("Collecting SNR data...")
    M = 8
    K = 8
    array = UniformLinearArray(M=M, d=0.5)
    T = 128
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_trials = 10
    true_doas = np.radians(np.linspace(-50, 50, K))

    snr_values = [-10, -5, 0, 5, 10, 15, 20]
    alg_names = ['MUSIC', 'COP', 'T-COP(1)', 'T-COP(5)', 'T-COP(10)']

    results = {name: {'pd': [], 'rmse': []} for name in alg_names}

    for snr_db in snr_values:
        print(f"  SNR={snr_db}dB...", end=' ', flush=True)

        for name in alg_names:
            pds, rmses = [], []
            for trial in range(n_trials):
                np.random.seed(trial * 100)
                X, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                             "non_stationary")
                alg = make_alg(name, array, K, snr_db)
                pd, rmse = run_trial(alg, X, scan_angles, true_doas,
                                     n_scans=get_n_scans(name), snr_db=snr_db)
                pds.append(pd)
                rmses.append(rmse)

            results[name]['pd'].append(np.mean(pds))
            results[name]['rmse'].append(np.mean(rmses))

        print("done")

    return snr_values, results


def collect_resolution():
    """Collect close-spacing resolution data."""
    print("Collecting resolution data...")
    M = 8
    K = 3
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 512
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_trials = 10

    spacing_values = [15, 10, 7, 5, 3, 2, 1]
    alg_names = ['MUSIC', 'Capon', 'COP', 'T-COP(5)']

    results = {name: {'pd': [], 'rmse': []} for name in alg_names}

    for spacing in spacing_values:
        true_doas = np.radians([0 - spacing, 0, 0 + spacing])
        print(f"  spacing={spacing} deg...", end=' ', flush=True)

        for name in alg_names:
            pds, rmses = [], []
            for trial in range(n_trials):
                np.random.seed(trial * 100)
                X, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                             "non_stationary")
                alg = make_alg(name, array, K, snr_db)
                pd, rmse = run_trial(alg, X, scan_angles, true_doas,
                                     n_scans=get_n_scans(name), snr_db=snr_db)
                pds.append(pd)
                rmses.append(rmse)

            results[name]['pd'].append(np.mean(pds))
            results[name]['rmse'].append(np.mean(rmses))

        print("done")

    return spacing_values, results


def collect_snapshots():
    """Collect snapshot efficiency data."""
    print("Collecting snapshot data...")
    M = 8
    K = 6
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 10
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_trials = 10
    true_doas = np.radians(np.linspace(-40, 40, K))

    T_values = [32, 64, 128, 256, 512, 1024]
    alg_names = ['MUSIC', 'COP', 'T-COP(5)']

    results = {name: {'pd': [], 'rmse': []} for name in alg_names}

    for T in T_values:
        print(f"  T={T}...", end=' ', flush=True)

        for name in alg_names:
            pds, rmses = [], []
            for trial in range(n_trials):
                np.random.seed(trial * 100)
                X, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                             "non_stationary")
                alg = make_alg(name, array, K, snr_db)
                pd, rmse = run_trial(alg, X, scan_angles, true_doas,
                                     n_scans=get_n_scans(name), snr_db=snr_db)
                pds.append(pd)
                rmses.append(rmse)

            results[name]['pd'].append(np.mean(pds))
            results[name]['rmse'].append(np.mean(rmses))

        print("done")

    return T_values, results


# ============================================================
# Plotting Functions
# ============================================================

def plot_k_scaling(K_values, results):
    """Fig 1 & 2: K Scaling with underdetermined region annotation."""
    M = 8
    max_conv = M - 1   # Conventional limit (MUSIC/ESPRIT/Capon)
    max_cop = 14        # COP limit = rho*(M-1) for rho=2

    for metric, ylabel, title_suffix, ylim, loc, fname in [
        ('pd', 'Detection Rate (Pd)',
         'Detection Rate vs Source Count', [-0.05, 1.05], 'lower left',
         'fig1_k_scaling_pd.png'),
        ('rmse', 'RMSE (degrees)',
         'RMSE vs Source Count', None, 'upper left',
         'fig2_k_scaling_rmse.png'),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5))

        # Shaded regions
        ax.axvspan(min(K_values) - 0.5, max_conv + 0.5,
                   alpha=0.08, color='green', zorder=0)
        ax.axvspan(max_conv + 0.5, max_cop + 0.5,
                   alpha=0.08, color='blue', zorder=0)

        # Vertical boundary lines
        ax.axvline(x=max_conv, color='green', linestyle='--',
                   alpha=0.6, linewidth=1.5)
        ax.axvline(x=M, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        ax.axvline(x=max_cop, color='blue', linestyle='--',
                   alpha=0.6, linewidth=1.5)

        # Region labels
        y_label = 1.0 if metric == 'pd' else max(
            max(d[metric]) for d in results.values()) * 0.95

        ax.text((min(K_values) + max_conv) / 2, y_label,
                'Determined\n(K < M)',
                fontsize=8, ha='center', va='top',
                color='green', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8, edgecolor='green'))

        ax.text((max_conv + max_cop) / 2 + 0.5, y_label,
                'Underdetermined\n(M-1 < K < COP limit)',
                fontsize=8, ha='center', va='top',
                color='blue', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8, edgecolor='blue'))

        # Boundary annotations
        if metric == 'pd':
            ax.annotate(f'K=M-1={max_conv}\nConventional limit',
                       xy=(max_conv, 0.15), fontsize=8, color='green',
                       ha='center', va='bottom')
            ax.annotate(f'K={max_cop}\nCOP limit\n(rho*(M-1))',
                       xy=(max_cop, 0.15), fontsize=8, color='blue',
                       ha='center', va='bottom')

        # Plot data
        for name, data in results.items():
            s = ALG_STYLES[name]
            ax.plot(K_values, data[metric], color=s[0], marker=s[1],
                    linestyle=s[2], label=s[3], linewidth=2, markersize=7)

        ax.set_xlabel('Number of Sources (K)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Underdetermined DOA: {title_suffix}\n'
                     f'M={M} sensors, SNR=15dB, T=256 snapshots')
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xticks(K_values)
        ax.legend(loc=loc, framealpha=0.95, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, fname))
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_snr(snr_values, results):
    """Fig 3 & 4: SNR Robustness."""
    # Fig 3: Pd vs SNR
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in results.items():
        s = ALG_STYLES[name]
        ax.plot(snr_values, data['pd'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3])

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Detection Rate (Pd)')
    ax.set_title('SNR Robustness: Detection Rate\n'
                 'M=8 sensors, K=8 sources (K>M-1: Underdetermined), T=128')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='lower right', framealpha=0.9, fontsize=8)
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_snr_pd.png'))
    plt.close(fig)
    print(f"  Saved fig3_snr_pd.png")

    # Fig 4: RMSE vs SNR
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in results.items():
        s = ALG_STYLES[name]
        ax.plot(snr_values, data['rmse'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3])

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('RMSE (degrees)')
    ax.set_title('SNR Robustness: RMSE\n'
                 'M=8 sensors, K=8 sources (K>M-1: Underdetermined), T=128')
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_snr_rmse.png'))
    plt.close(fig)
    print(f"  Saved fig4_snr_rmse.png")


def plot_resolution(spacing_values, results):
    """Fig 5: Close-spacing resolution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, data in results.items():
        s = ALG_STYLES[name]
        ax1.plot(spacing_values, data['pd'], color=s[0], marker=s[1],
                 linestyle=s[2], label=s[3])
        ax2.plot(spacing_values, data['rmse'], color=s[0], marker=s[1],
                 linestyle=s[2], label=s[3])

    ax1.set_xlabel('Source Spacing (degrees)')
    ax1.set_ylabel('Detection Rate (Pd)')
    ax1.set_title('Close-Spacing Resolution: Pd')
    ax1.set_ylim([-0.05, 1.05])
    ax1.invert_xaxis()
    ax1.legend(loc='lower left', framealpha=0.9)

    ax2.set_xlabel('Source Spacing (degrees)')
    ax2.set_ylabel('RMSE (degrees)')
    ax2.set_title('Close-Spacing Resolution: RMSE')
    ax2.invert_xaxis()
    ax2.legend(loc='upper left', framealpha=0.9)

    fig.suptitle('Close-Spacing Resolution (M=8, K=3, SNR=15dB, T=512)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_resolution.png'))
    plt.close(fig)
    print(f"  Saved fig5_resolution.png")


def plot_snapshots(T_values, results):
    """Fig 6: Snapshot efficiency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, data in results.items():
        s = ALG_STYLES[name]
        ax1.semilogx(T_values, data['pd'], color=s[0], marker=s[1],
                     linestyle=s[2], label=s[3], base=2)
        ax2.semilogx(T_values, data['rmse'], color=s[0], marker=s[1],
                     linestyle=s[2], label=s[3], base=2)

    ax1.set_xlabel('Number of Snapshots (T)')
    ax1.set_ylabel('Detection Rate (Pd)')
    ax1.set_title('Snapshot Efficiency: Pd')
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xticks(T_values)
    ax1.set_xticklabels([str(t) for t in T_values])
    ax1.legend(loc='lower right', framealpha=0.9)

    ax2.set_xlabel('Number of Snapshots (T)')
    ax2.set_ylabel('RMSE (degrees)')
    ax2.set_title('Snapshot Efficiency: RMSE')
    ax2.set_xticks(T_values)
    ax2.set_xticklabels([str(t) for t in T_values])
    ax2.legend(loc='upper right', framealpha=0.9)

    fig.suptitle('Snapshot Efficiency (M=8, K=6, SNR=10dB)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_snapshots.png'))
    plt.close(fig)
    print(f"  Saved fig6_snapshots.png")


def plot_combined_summary(k_data, snr_data, res_data, snap_data):
    """Create a combined 2x3 summary figure."""
    K_values, k_results = k_data
    snr_values, snr_results = snr_data
    spacing_values, res_results = res_data
    T_values, snap_results = snap_data

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) K scaling Pd
    ax = axes[0, 0]
    ax.axvspan(7.5, 14.5, alpha=0.08, color='blue', zorder=0)
    ax.axvline(x=7, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=14, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    for name, data in k_results.items():
        s = ALG_STYLES[name]
        ax.plot(K_values, data['pd'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3], markersize=5)
    ax.set_xlabel('K (sources)')
    ax.set_ylabel('Pd')
    ax.set_title('(a) K Scaling: Detection Rate')
    ax.set_ylim([-0.05, 1.05])
    ax.text(5, 0.08, 'K<M-1', fontsize=7, color='green')
    ax.text(10, 0.08, 'Underdetermined', fontsize=7, color='blue')
    ax.legend(fontsize=6, loc='lower left')

    # (0,1) SNR Pd
    ax = axes[0, 1]
    for name, data in snr_results.items():
        s = ALG_STYLES[name]
        ax.plot(snr_values, data['pd'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3], markersize=5)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Pd')
    ax.set_title('(b) SNR Robustness: Detection Rate')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=7, loc='lower right')

    # (0,2) Resolution Pd
    ax = axes[0, 2]
    for name, data in res_results.items():
        s = ALG_STYLES[name]
        ax.plot(spacing_values, data['pd'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3], markersize=5)
    ax.set_xlabel('Source Spacing (deg)')
    ax.set_ylabel('Pd')
    ax.set_title('(c) Resolution: Detection Rate')
    ax.set_ylim([-0.05, 1.05])
    ax.invert_xaxis()
    ax.legend(fontsize=7, loc='lower left')

    # (1,0) K scaling RMSE
    ax = axes[1, 0]
    ax.axvspan(7.5, 14.5, alpha=0.08, color='blue', zorder=0)
    ax.axvline(x=7, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=14, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    for name, data in k_results.items():
        s = ALG_STYLES[name]
        ax.plot(K_values, data['rmse'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3], markersize=5)
    ax.set_xlabel('K (sources)')
    ax.set_ylabel('RMSE (deg)')
    ax.set_title('(d) K Scaling: RMSE')
    ax.legend(fontsize=6, loc='upper left')

    # (1,1) SNR RMSE
    ax = axes[1, 1]
    for name, data in snr_results.items():
        s = ALG_STYLES[name]
        ax.plot(snr_values, data['rmse'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3], markersize=5)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('RMSE (deg)')
    ax.set_title('(e) SNR Robustness: RMSE')
    ax.set_yscale('log')
    ax.legend(fontsize=7, loc='upper right')

    # (1,2) Snapshots RMSE
    ax = axes[1, 2]
    for name, data in snap_results.items():
        s = ALG_STYLES[name]
        ax.semilogx(T_values, data['rmse'], color=s[0], marker=s[1],
                     linestyle=s[2], label=s[3], markersize=5, base=2)
    ax.set_xlabel('Snapshots (T)')
    ax.set_ylabel('RMSE (deg)')
    ax.set_title('(f) Snapshot Efficiency: RMSE')
    ax.set_xticks(T_values)
    ax.set_xticklabels([str(t) for t in T_values])
    ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('COP Family Algorithm Performance Summary\n'
                 'Base: 2rho-th Order Subspace COP (Choi & Yoo, IEEE TSP 2015)',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_summary_combined.png'))
    plt.close(fig)
    print(f"  Saved fig_summary_combined.png")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating Benchmark Plots")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Collect data
    k_data = collect_k_scaling()
    snr_data = collect_snr()
    res_data = collect_resolution()
    snap_data = collect_snapshots()

    # Generate individual plots
    print("\nGenerating plots...")
    plot_k_scaling(*k_data)
    plot_snr(*snr_data)
    plot_resolution(*res_data)
    plot_snapshots(*snap_data)

    # Generate combined summary
    plot_combined_summary(k_data, snr_data, res_data, snap_data)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("Done!")
