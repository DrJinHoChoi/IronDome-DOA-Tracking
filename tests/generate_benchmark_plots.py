#!/usr/bin/env python3
"""Generate publication-quality benchmark plots for COP family algorithms.

Generates 8 figures:
  Fig 1: K Scaling - Pd vs K (underdetermined capability, K=3~20)
  Fig 2: K Scaling - RMSE vs K
  Fig 3: SNR Robustness - Pd vs SNR (includes SD-COP)
  Fig 4: SNR Robustness - RMSE vs SNR
  Fig 5: Close-Spacing Resolution - Pd vs spacing
  Fig 6: Snapshot Efficiency - RMSE vs T
  Fig 11: Extended K Scaling (K=14~25) - SD-COP vs COP capacity limit
  Fig 12: SD-COP Deflation Stage Analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import (SubspaceCOP, TemporalCOP, SequentialDeflationCOP,
                                MUSIC, ESPRIT, Capon, COP_CBF, COP_MVDR)
from iron_dome_sim.eval.metrics import rmse_doa, detection_rate
from iron_dome_sim.eval.crlb import crlb_rmse, crlb_stochastic, crlb_cop

# ============================================================
# Style configuration
# ============================================================
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

# Algorithm styles: (color, marker, linestyle, label)
# M=8 sensors -> conventional max K=M-1=7, COP max K=rho*(M-1)=14
ALG_STYLES = {
    'MUSIC':     ('#AAAAAA', 's', '--', 'MUSIC'),
    'ESPRIT':    ('#AA8800', 'D', '-.', 'ESPRIT'),
    'Capon':     ('#008888', '^', ':',  'Capon'),
    'COP-CBF':   ('#66BB66', 'h', '--', 'COP-CBF (K-free)'),
    'COP-MVDR':  ('#CC4400', 'p', '-',  'COP-MVDR (K-free) [Proposed]'),
    'COP':       ('#0055CC', 'o', '-',  'COP-4th [Proposed]'),
    'T-COP(1)':  ('#FF8800', 'v', ':',  'T-COP, 1 scan'),
    'T-COP(5)':  ('#DD0000', 'P', '-',  'T-COP, 5 scans [Proposed]'),
    'T-COP(10)': ('#8800CC', '*', '-',  'T-COP, 10 scans [Proposed]'),
    'SD-COP':    ('#00AA00', 'X', '-.',  'SD-COP [Proposed]'),
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
    elif name == 'COP-CBF':
        return COP_CBF(array, num_sources=K, rho=2)
    elif name == 'COP-MVDR':
        return COP_MVDR(array, num_sources=K, rho=2)
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
    """Collect K scaling data - EXTENDED to K=20 for SD-COP."""
    print("Collecting K scaling data (extended to K=20)...")
    M = 8
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 256
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_trials = 10

    # Extended K range: 3 to 20 (beyond COP limit of 14)
    K_values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    alg_names = ['MUSIC', 'ESPRIT', 'Capon', 'COP-CBF', 'COP-MVDR', 'COP', 'T-COP(5)', 'SD-COP']

    results = {name: {'pd': [], 'rmse': []} for name in alg_names}

    for K in K_values:
        true_doas = np.radians(np.linspace(-60, 60, K))
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
    """Collect SNR robustness data - NOW INCLUDES SD-COP."""
    print("Collecting SNR data (with SD-COP)...")
    M = 8
    K = 8
    array = UniformLinearArray(M=M, d=0.5)
    T = 128
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_trials = 10
    true_doas = np.radians(np.linspace(-50, 50, K))

    snr_values = [-10, -5, 0, 5, 10, 15, 20]
    alg_names = ['MUSIC', 'COP-CBF', 'COP-MVDR', 'COP', 'T-COP(1)', 'T-COP(5)', 'T-COP(10)', 'SD-COP']

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
    """Collect close-spacing resolution data - NOW INCLUDES SD-COP."""
    print("Collecting resolution data (with SD-COP)...")
    M = 8
    K = 3
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 512
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_trials = 10

    spacing_values = [15, 10, 7, 5, 3, 2, 1]
    alg_names = ['MUSIC', 'Capon', 'COP-CBF', 'COP-MVDR', 'COP', 'T-COP(5)', 'SD-COP']

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
    """Collect snapshot efficiency data - NOW INCLUDES SD-COP."""
    print("Collecting snapshot data (with SD-COP)...")
    M = 8
    K = 6
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 10
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_trials = 10
    true_doas = np.radians(np.linspace(-40, 40, K))

    T_values = [32, 64, 128, 256, 512, 1024]
    alg_names = ['MUSIC', 'COP-CBF', 'COP-MVDR', 'COP', 'T-COP(5)', 'SD-COP']

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


def collect_extended_k():
    """Collect EXTENDED K scaling specifically for SD-COP (K=10~25).

    This benchmark focuses on the super-underdetermined regime where
    K exceeds the single-stage COP capacity rho*(M-1)=14.
    """
    print("Collecting extended K data (K=10~25, SD-COP focus)...")
    M = 8
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 512  # More snapshots for harder scenarios
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_trials = 8

    K_values = [10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24]
    alg_names = ['COP', 'SD-COP', 'T-COP(5)']

    results = {name: {'pd': [], 'rmse': []} for name in alg_names}

    for K in K_values:
        true_doas = np.radians(np.linspace(-65, 65, K))
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


def collect_sdcop_stages():
    """Collect SD-COP deflation stage analysis data.

    Shows how many deflation stages SD-COP uses and per-stage
    performance for varying K.
    """
    print("Collecting SD-COP stage analysis data...")
    M = 8
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 512
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)

    K_values = [8, 10, 12, 14, 16, 18, 20, 22]
    results = {'K': K_values, 'n_stages': [], 'pd': [], 'rmse': [],
               'stage_doas': [], 'energy_ratios': []}

    for K in K_values:
        true_doas = np.radians(np.linspace(-60, 60, K))
        print(f"  K={K}...", end=' ', flush=True)

        np.random.seed(42)
        X, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                     "non_stationary")
        alg = SequentialDeflationCOP(array, rho=2, num_sources=K)
        doa_est, _ = alg.estimate(X, scan_angles)

        n_stages = len(alg.stage_results)
        stage_n_doas = [s['n_detected'] for s in alg.stage_results]
        energy_ratios = [s['energy_ratio'] for s in alg.stage_results]

        rmse_val, _ = rmse_doa(doa_est, true_doas)
        pd, _ = detection_rate(doa_est, true_doas)

        results['n_stages'].append(n_stages)
        results['pd'].append(pd)
        results['rmse'].append(np.degrees(rmse_val))
        results['stage_doas'].append(stage_n_doas)
        results['energy_ratios'].append(energy_ratios)

        print(f"stages={n_stages}, Pd={pd:.2f}, RMSE={np.degrees(rmse_val):.1f}")

    return results


# ============================================================
# Plotting Functions
# ============================================================

def plot_k_scaling(K_values, results):
    """Fig 1 & 2: K Scaling with THREE regions (determined/underdetermined/super-underdetermined)."""
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
        fig, ax = plt.subplots(figsize=(16, 9))

        # THREE shaded regions
        ax.axvspan(min(K_values) - 0.5, max_conv + 0.5,
                   alpha=0.06, color='green', zorder=0)
        ax.axvspan(max_conv + 0.5, max_cop + 0.5,
                   alpha=0.06, color='blue', zorder=0)
        ax.axvspan(max_cop + 0.5, max(K_values) + 0.5,
                   alpha=0.06, color='red', zorder=0)

        # Vertical boundary lines
        ax.axvline(x=max_conv, color='green', linestyle='--',
                   alpha=0.6, linewidth=2.0)
        ax.axvline(x=max_cop, color='blue', linestyle='--',
                   alpha=0.6, linewidth=2.0)

        # Region labels
        y_label = 1.0 if metric == 'pd' else max(
            max(d[metric]) for d in results.values()) * 0.95

        ax.text((min(K_values) + max_conv) / 2, y_label,
                'Determined\n(K < M)',
                fontsize=14, ha='center', va='top',
                color='green', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8, edgecolor='green'))

        ax.text((max_conv + max_cop) / 2 + 0.5, y_label,
                'Underdetermined\n(M-1 < K \u2264 COP limit)',
                fontsize=14, ha='center', va='top',
                color='blue', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8, edgecolor='blue'))

        ax.text((max_cop + max(K_values)) / 2, y_label,
                'Super-Underdetermined\n(K > COP limit)\nSD-COP only',
                fontsize=14, ha='center', va='top',
                color='red', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8, edgecolor='red'))

        # Boundary annotations
        if metric == 'pd':
            ax.annotate(f'K=M-1={max_conv}\nConventional limit',
                       xy=(max_conv, 0.15), fontsize=13, color='green',
                       ha='center', va='bottom')
            ax.annotate(f'K={max_cop}\nCOP limit\n(\u03c1(M-1))',
                       xy=(max_cop, 0.15), fontsize=13, color='blue',
                       ha='center', va='bottom')

        # Plot data
        for name, data in results.items():
            s = ALG_STYLES[name]
            ax.plot(K_values, data[metric], color=s[0], marker=s[1],
                    linestyle=s[2], label=s[3], linewidth=2.5, markersize=10)

        ax.set_xlabel('Number of Sources (K)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Underdetermined DOA: {title_suffix}\n'
                     f'M={M} sensors, SNR=15dB, T=256 snapshots',
                     fontweight='bold')
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xticks(K_values[::2])
        ax.legend(loc=loc, framealpha=0.95)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, fname))
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_snr(snr_values, results):
    """Fig 3 & 4: SNR Robustness (now includes SD-COP)."""
    # Fig 3: Pd vs SNR
    fig, ax = plt.subplots(figsize=(16, 8))
    for name, data in results.items():
        s = ALG_STYLES[name]
        ax.plot(snr_values, data['pd'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3])

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Detection Rate (Pd)')
    ax.set_title('SNR Robustness: Detection Rate\n'
                 'M=8 sensors, K=8 sources (K>M-1: Underdetermined), T=128',
                 fontweight='bold')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='lower right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_snr_pd.png'))
    plt.close(fig)
    print(f"  Saved fig3_snr_pd.png")

    # Fig 4: RMSE vs SNR with exact CRLB
    fig, ax = plt.subplots(figsize=(16, 8))
    for name, data in results.items():
        s = ALG_STYLES[name]
        ax.plot(snr_values, data['rmse'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3])

    # Add exact CRLB curves
    snr_fine = np.array(snr_values, dtype=float)
    true_doas = np.radians(np.linspace(-50, 50, 8))  # K=8
    crlb_std = crlb_rmse(true_doas, 8, snr_fine, 128, rho=1)
    crlb_ho = crlb_rmse(true_doas, 8, snr_fine, 128, rho=2)
    ax.plot(snr_values, crlb_std, 'k--', linewidth=2.0, alpha=0.7,
            label='CRLB (standard)')
    ax.plot(snr_values, crlb_ho, 'k:', linewidth=2.0, alpha=0.7,
            label='CRLB (4th-order)')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('RMSE (degrees)')
    ax.set_title('SNR Robustness: RMSE\n'
                 'M=8 sensors, K=8 sources (K>M-1: Underdetermined), T=128',
                 fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_snr_rmse.png'))
    plt.close(fig)
    print(f"  Saved fig4_snr_rmse.png")


def plot_resolution(spacing_values, results):
    """Fig 5: Close-spacing resolution (now includes SD-COP)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

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
                 fontsize=22, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_resolution.png'))
    plt.close(fig)
    print(f"  Saved fig5_resolution.png")


def plot_snapshots(T_values, results):
    """Fig 6: Snapshot efficiency (now includes SD-COP + CRLB)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

    for name, data in results.items():
        s = ALG_STYLES[name]
        ax1.semilogx(T_values, data['pd'], color=s[0], marker=s[1],
                     linestyle=s[2], label=s[3], base=2)
        ax2.semilogx(T_values, data['rmse'], color=s[0], marker=s[1],
                     linestyle=s[2], label=s[3], base=2)

    # Add exact CRLB curves to RMSE plot (ax2)
    true_doas = np.radians(np.linspace(-40, 40, 6))  # K=6, same as collect_snapshots
    crlb_std_vals = []
    crlb_ho_vals = []
    for T in T_values:
        crb, _ = crlb_stochastic(true_doas, 8, 10, T)  # M=8, SNR=10dB
        mean_crb = np.mean(crb)
        crlb_std_vals.append(np.degrees(np.sqrt(mean_crb)) if mean_crb < np.inf else np.inf)
        crb, _ = crlb_cop(true_doas, 8, 10, T, rho=2)
        mean_crb = np.mean(crb)
        crlb_ho_vals.append(np.degrees(np.sqrt(mean_crb)) if mean_crb < np.inf else np.inf)

    ax2.semilogx(T_values, crlb_std_vals, 'k--', linewidth=2.0, alpha=0.7,
                 label='CRLB (standard)', base=2)
    ax2.semilogx(T_values, crlb_ho_vals, 'k:', linewidth=2.0, alpha=0.7,
                 label='CRLB (4th-order)', base=2)

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
                 fontsize=22, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_snapshots.png'))
    plt.close(fig)
    print(f"  Saved fig6_snapshots.png")


def plot_extended_k(K_values, results):
    """Fig 11: Extended K scaling - SD-COP beyond COP capacity limit."""
    max_cop = 14

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

    for ax, metric, ylabel, ylim in [
        (ax1, 'pd', 'Detection Rate (Pd)', [-0.05, 1.05]),
        (ax2, 'rmse', 'RMSE (degrees)', None),
    ]:
        # Shaded regions
        ax.axvspan(min(K_values) - 0.5, max_cop + 0.5,
                   alpha=0.06, color='blue', zorder=0)
        ax.axvspan(max_cop + 0.5, max(K_values) + 0.5,
                   alpha=0.06, color='red', zorder=0)
        ax.axvline(x=max_cop, color='blue', linestyle='--',
                   alpha=0.7, linewidth=2.5)

        # Region labels
        if metric == 'pd':
            ax.text(12, 0.08, 'COP\nCapacity',
                    fontsize=15, ha='center', color='blue', fontstyle='italic')
            ax.text(19, 0.08, 'Beyond COP Limit\n(SD-COP deflation)',
                    fontsize=15, ha='center', color='red', fontstyle='italic')

        for name, data in results.items():
            s = ALG_STYLES[name]
            ax.plot(K_values, data[metric], color=s[0], marker=s[1],
                    linestyle=s[2], label=s[3], linewidth=2.5, markersize=10)

        ax.set_xlabel('Number of Sources (K)')
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xticks(K_values[::2])
        ax.legend(loc='lower left' if metric == 'pd' else 'upper left',
                  framealpha=0.95)

    fig.suptitle('Extended K Scaling: SD-COP Beyond COP Capacity Limit\n'
                 'M=8 sensors, SNR=15dB, T=512 | COP limit = \u03c1(M-1) = 14',
                 fontsize=22, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig11_extended_k.png'))
    plt.close(fig)
    print(f"  Saved fig11_extended_k.png")


def plot_sdcop_stages(stage_data):
    """Fig 12: SD-COP deflation stage analysis."""
    K_values = stage_data['K']
    n_stages = stage_data['n_stages']
    pd_vals = stage_data['pd']
    rmse_vals = stage_data['rmse']
    stage_doas = stage_data['stage_doas']

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (0,0) Number of deflation stages vs K
    ax = axes[0, 0]
    colors = ['#2ca02c' if k <= 14 else '#d62728' for k in K_values]
    ax.bar(K_values, n_stages, color=colors, edgecolor='black', alpha=0.7, width=1.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Single stage (no deflation)')
    ax.axvline(x=14, color='blue', linestyle='--', alpha=0.7, linewidth=2,
               label='COP limit (K=14)')
    ax.set_xlabel('Number of Sources (K)')
    ax.set_ylabel('Number of Deflation Stages')
    ax.set_title('(a) Deflation Stages Required')
    ax.set_xticks(K_values)
    ax.legend(fontsize=8)

    # (0,1) DOAs detected per stage (stacked bar)
    ax = axes[0, 1]
    max_n_stages = max(n_stages)
    bottoms = [0] * len(K_values)
    stage_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for s_idx in range(max_n_stages):
        heights = []
        for i, sd in enumerate(stage_doas):
            if s_idx < len(sd):
                heights.append(sd[s_idx])
            else:
                heights.append(0)
        label = f'Stage {s_idx+1}'
        c = stage_colors[s_idx % len(stage_colors)]
        ax.bar(K_values, heights, bottom=bottoms, color=c, edgecolor='black',
               alpha=0.7, width=1.5, label=label)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    # True K reference line
    ax.plot(K_values, K_values, 'k--', linewidth=1.5, label='True K', alpha=0.6)
    ax.axvline(x=14, color='blue', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Number of Sources (K)')
    ax.set_ylabel('DOAs Detected')
    ax.set_title('(b) DOAs Detected per Stage')
    ax.set_xticks(K_values)
    ax.legend(fontsize=7, loc='upper left')

    # (1,0) Pd vs K
    ax = axes[1, 0]
    ax.plot(K_values, pd_vals, 'o-', color='#2ca02c', linewidth=2.5, markersize=8,
            label='SD-COP-4th')
    ax.axvline(x=14, color='blue', linestyle='--', alpha=0.7, linewidth=2,
               label='COP limit (K=14)')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of Sources (K)')
    ax.set_ylabel('Detection Rate (Pd)')
    ax.set_title('(c) SD-COP Detection Rate')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xticks(K_values)
    ax.legend(fontsize=9)

    # (1,1) RMSE vs K
    ax = axes[1, 1]
    ax.plot(K_values, rmse_vals, 's-', color='#2ca02c', linewidth=2.5, markersize=8,
            label='SD-COP-4th')
    ax.axvline(x=14, color='blue', linestyle='--', alpha=0.7, linewidth=2,
               label='COP limit (K=14)')
    ax.set_xlabel('Number of Sources (K)')
    ax.set_ylabel('RMSE (degrees)')
    ax.set_title('(d) SD-COP RMSE')
    ax.set_xticks(K_values)
    ax.legend(fontsize=9)

    fig.suptitle('SD-COP Deflation Stage Analysis\n'
                 'M=8 sensors, SNR=15dB, T=512 | Sequential deflation in HOC domain',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig12_sdcop_stages.png'))
    plt.close(fig)
    print(f"  Saved fig12_sdcop_stages.png")


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
    ax.axvspan(14.5, max(K_values) + 0.5, alpha=0.06, color='red', zorder=0)
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
    ax.text(10.5, 0.08, 'Underdetermined', fontsize=7, color='blue')
    ax.text(17, 0.08, 'SD-COP', fontsize=7, color='red')
    ax.legend(fontsize=5.5, loc='lower left')

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
    ax.legend(fontsize=6, loc='lower right')

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
    ax.axvspan(14.5, max(K_values) + 0.5, alpha=0.06, color='red', zorder=0)
    ax.axvline(x=7, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=14, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    for name, data in k_results.items():
        s = ALG_STYLES[name]
        ax.plot(K_values, data['rmse'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3], markersize=5)
    ax.set_xlabel('K (sources)')
    ax.set_ylabel('RMSE (deg)')
    ax.set_title('(d) K Scaling: RMSE')
    ax.legend(fontsize=5.5, loc='upper left')

    # (1,1) SNR RMSE with CRLB
    ax = axes[1, 1]
    for name, data in snr_results.items():
        s = ALG_STYLES[name]
        ax.plot(snr_values, data['rmse'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3], markersize=5)
    # CRLB for SNR plot: M=8, K=8, T=128
    snr_fine = np.array(snr_values, dtype=float)
    snr_doas = np.radians(np.linspace(-50, 50, 8))
    crlb_std_snr = crlb_rmse(snr_doas, 8, snr_fine, 128, rho=1)
    crlb_ho_snr = crlb_rmse(snr_doas, 8, snr_fine, 128, rho=2)
    ax.plot(snr_values, crlb_std_snr, 'k--', linewidth=1.0, alpha=0.6,
            label='CRLB (std)', markersize=3)
    ax.plot(snr_values, crlb_ho_snr, 'k:', linewidth=1.0, alpha=0.6,
            label='CRLB (4th)', markersize=3)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('RMSE (deg)')
    ax.set_title('(e) SNR Robustness: RMSE')
    ax.set_yscale('log')
    ax.legend(fontsize=5.5, loc='upper right')

    # (1,2) Snapshots RMSE with CRLB
    ax = axes[1, 2]
    for name, data in snap_results.items():
        s = ALG_STYLES[name]
        ax.semilogx(T_values, data['rmse'], color=s[0], marker=s[1],
                     linestyle=s[2], label=s[3], markersize=5, base=2)
    # CRLB for snapshot plot: M=8, K=6, SNR=10dB
    snap_doas = np.radians(np.linspace(-40, 40, 6))
    crlb_std_snap = []
    crlb_ho_snap = []
    for T in T_values:
        crb, _ = crlb_stochastic(snap_doas, 8, 10, T)
        mean_crb = np.mean(crb)
        crlb_std_snap.append(np.degrees(np.sqrt(mean_crb)) if mean_crb < np.inf else np.inf)
        crb, _ = crlb_cop(snap_doas, 8, 10, T, rho=2)
        mean_crb = np.mean(crb)
        crlb_ho_snap.append(np.degrees(np.sqrt(mean_crb)) if mean_crb < np.inf else np.inf)
    ax.semilogx(T_values, crlb_std_snap, 'k--', linewidth=1.0, alpha=0.6,
                label='CRLB (std)', base=2, markersize=3)
    ax.semilogx(T_values, crlb_ho_snap, 'k:', linewidth=1.0, alpha=0.6,
                label='CRLB (4th)', base=2, markersize=3)
    ax.set_xlabel('Snapshots (T)')
    ax.set_ylabel('RMSE (deg)')
    ax.set_title('(f) Snapshot Efficiency: RMSE')
    ax.set_xticks(T_values)
    ax.set_xticklabels([str(t) for t in T_values])
    ax.legend(fontsize=5.5, loc='upper right')

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
    print("Generating Benchmark Plots (with SD-COP)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Collect data
    k_data = collect_k_scaling()
    snr_data = collect_snr()
    res_data = collect_resolution()
    snap_data = collect_snapshots()
    ext_k_data = collect_extended_k()
    stage_data = collect_sdcop_stages()

    # Generate individual plots
    print("\nGenerating plots...")
    plot_k_scaling(*k_data)
    plot_snr(*snr_data)
    plot_resolution(*res_data)
    plot_snapshots(*snap_data)
    plot_extended_k(*ext_k_data)
    plot_sdcop_stages(stage_data)

    # Generate combined summary
    plot_combined_summary(k_data, snr_data, res_data, snap_data)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("Done!")
