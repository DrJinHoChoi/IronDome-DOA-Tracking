#!/usr/bin/env python3
"""Benchmark for M=4 sensors — practical small-array scenario.

M=4, rho=2 → M_v = 2*(4-1)+1 = 7
  Conventional limit: K ≤ M-1 = 3
  COP limit:          K ≤ rho*(M-1) = 6
  SD-COP:             K > 6

Demonstrates COP advantage with minimal hardware (e.g., Cortex-M7 + 4-mic array).
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
from iron_dome_sim.doa import (SubspaceCOP, TemporalCOP, SequentialDeflationCOP,
                                MUSIC, ESPRIT, Capon, COP_CBF, COP_MVDR)
from iron_dome_sim.eval.metrics import rmse_doa, detection_rate

# ============================================================
# Configuration
# ============================================================
M = 4
RHO = 2
M_V = RHO * (M - 1) + 1  # = 7
MAX_CONV = M - 1           # = 3
MAX_COP = RHO * (M - 1)   # = 6

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'legend.fontsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 2.5,
    'lines.markersize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

ALG_STYLES = {
    'MUSIC':     ('#AAAAAA', 's', '--', 'MUSIC'),
    'ESPRIT':    ('#AA8800', 'D', '-.', 'ESPRIT'),
    'Capon':     ('#008888', '^', ':',  'Capon'),
    'COP-CBF':   ('#66BB66', 'h', '--', 'COP-CBF (K-free)'),
    'COP-MVDR':  ('#CC4400', 'p', '-',  'COP-MVDR (K-free) [Proposed]'),
    'COP':       ('#0055CC', 'o', '-',  'COP-4th [Proposed]'),
    'T-COP(5)':  ('#DD0000', 'P', '-',  'T-COP, 5 scans [Proposed]'),
    'SD-COP':    ('#00AA00', 'X', '-.',  'SD-COP [Proposed]'),
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', 'figures_m4')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def make_alg(name, array, K, snr_db=15):
    if name == 'MUSIC':
        return MUSIC(array, num_sources=min(K, M - 1))
    elif name == 'ESPRIT':
        return ESPRIT(array, num_sources=min(K, M - 1))
    elif name == 'Capon':
        return Capon(array, num_sources=min(K, M - 1))
    elif name == 'COP':
        return SubspaceCOP(array, rho=RHO, num_sources=K, spectrum_type="combined")
    elif name.startswith('T-COP'):
        return TemporalCOP(array, rho=RHO, num_sources=K, alpha=0.85)
    elif name == 'SD-COP':
        return SequentialDeflationCOP(array, rho=RHO, num_sources=K)
    elif name == 'COP-CBF':
        return COP_CBF(array, num_sources=K, rho=RHO)
    elif name == 'COP-MVDR':
        return COP_MVDR(array, num_sources=K, rho=RHO)
    raise ValueError(f"Unknown: {name}")


def get_n_scans(name):
    if name.startswith('T-COP('):
        return int(name.split('(')[1].rstrip(')'))
    return 1


def run_trial(alg, X, scan_angles, true_doas, n_scans=1, snr_db=15):
    try:
        if isinstance(alg, TemporalCOP) and n_scans > 1:
            M_loc, T = X.shape
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


def run_experiment(param_name, param_values, fixed_params, alg_names):
    """Generic experiment runner."""
    results = {name: {'pd': [], 'rmse': []} for name in alg_names}
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_trials = fixed_params.get('n_trials', 10)
    array = UniformLinearArray(M=M, d=0.5)

    for pval in param_values:
        print(f"  {param_name}={pval}...", end=' ', flush=True)

        # Set current param
        K = pval if param_name == 'K' else fixed_params['K']
        snr_db = pval if param_name == 'SNR' else fixed_params['snr_db']
        T = pval if param_name == 'T' else fixed_params['T']
        spacing = pval if param_name == 'spacing' else None

        if spacing is not None:
            true_doas = np.radians([0 - spacing, 0, 0 + spacing])
        elif param_name == 'K':
            true_doas = np.radians(np.linspace(-60, 60, K))
        else:
            true_doas = np.radians(np.linspace(-50, 50, K))

        for name in alg_names:
            pds, rmses = [], []
            for trial in range(n_trials):
                seed_val = abs(trial * 100 + (pval if isinstance(pval, int) else int(pval * 10))) + 1000
                np.random.seed(int(seed_val))
                X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")
                alg = make_alg(name, array, K, snr_db)
                pd, rmse = run_trial(alg, X, scan_angles, true_doas,
                                     n_scans=get_n_scans(name), snr_db=snr_db)
                pds.append(pd)
                rmses.append(rmse)
            results[name]['pd'].append(np.mean(pds))
            results[name]['rmse'].append(np.mean(rmses))

        print("done")

    return results


# ============================================================
# Experiments
# ============================================================
def exp_k_scaling():
    """Exp 1: K scaling — K=2..10 (conv limit=3, COP limit=6)."""
    print("\n" + "=" * 70)
    print(f"EXP 1: K Scaling (M={M}, SNR=15dB, T=256)")
    print(f"  Conventional limit: K={MAX_CONV}, COP limit: K={MAX_COP}")
    print("=" * 70)

    K_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    alg_names = ['MUSIC', 'ESPRIT', 'Capon', 'COP-CBF', 'COP-MVDR', 'COP', 'T-COP(5)', 'SD-COP']
    results = run_experiment('K', K_values, {'snr_db': 15, 'T': 256, 'n_trials': 10}, alg_names)
    return K_values, results


def exp_snr():
    """Exp 2: SNR robustness — K=4 (underdetermined for M=4)."""
    print("\n" + "=" * 70)
    print(f"EXP 2: SNR Robustness (M={M}, K=4, T=128)")
    print("=" * 70)

    snr_values = [-10, -5, 0, 5, 10, 15, 20]
    alg_names = ['MUSIC', 'COP-CBF', 'COP-MVDR', 'COP', 'T-COP(5)', 'SD-COP']
    results = run_experiment('SNR', snr_values, {'K': 4, 'T': 128, 'n_trials': 10}, alg_names)
    return snr_values, results


def exp_resolution():
    """Exp 3: Close-spacing resolution — K=3."""
    print("\n" + "=" * 70)
    print(f"EXP 3: Resolution (M={M}, K=3, SNR=15dB, T=512)")
    print("=" * 70)

    spacing_values = [15, 10, 7, 5, 3, 2]
    alg_names = ['MUSIC', 'Capon', 'COP-CBF', 'COP-MVDR', 'COP', 'T-COP(5)', 'SD-COP']
    results = run_experiment('spacing', spacing_values,
                             {'K': 3, 'snr_db': 15, 'T': 512, 'n_trials': 10}, alg_names)
    return spacing_values, results


def exp_snapshots():
    """Exp 4: Snapshot efficiency — K=4."""
    print("\n" + "=" * 70)
    print(f"EXP 4: Snapshots (M={M}, K=4, SNR=10dB)")
    print("=" * 70)

    T_values = [32, 64, 128, 256, 512, 1024]
    alg_names = ['MUSIC', 'COP-CBF', 'COP-MVDR', 'COP', 'T-COP(5)', 'SD-COP']
    results = run_experiment('T', T_values, {'K': 4, 'snr_db': 10, 'n_trials': 10}, alg_names)
    return T_values, results


# ============================================================
# Plotting
# ============================================================
def plot_k_scaling(K_values, results):
    for metric, ylabel, ylim, loc, fname in [
        ('pd', 'Detection Rate (Pd)', [-0.05, 1.05], 'lower left', 'fig1_m4_k_scaling_pd.png'),
        ('rmse', 'RMSE (degrees)', None, 'upper left', 'fig2_m4_k_scaling_rmse.png'),
    ]:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Three regions
        ax.axvspan(min(K_values) - 0.3, MAX_CONV + 0.5, alpha=0.07, color='green')
        ax.axvspan(MAX_CONV + 0.5, MAX_COP + 0.5, alpha=0.07, color='blue')
        ax.axvspan(MAX_COP + 0.5, max(K_values) + 0.3, alpha=0.07, color='red')
        ax.axvline(x=MAX_CONV, color='green', linestyle='--', alpha=0.6, linewidth=2)
        ax.axvline(x=MAX_COP, color='blue', linestyle='--', alpha=0.6, linewidth=2)

        y_label = 1.0 if metric == 'pd' else max(max(d[metric]) for d in results.values()) * 0.95
        ax.text((min(K_values) + MAX_CONV) / 2, y_label,
                'Determined\n(K ≤ 3)', fontsize=13, ha='center', va='top',
                color='green', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='green'))
        ax.text((MAX_CONV + MAX_COP) / 2 + 0.3, y_label,
                'Underdetermined\n(3 < K ≤ 6)', fontsize=13, ha='center', va='top',
                color='blue', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='blue'))
        ax.text((MAX_COP + max(K_values)) / 2, y_label,
                'Super-Underdetermined\n(K > 6) SD-COP only', fontsize=13, ha='center', va='top',
                color='red', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))

        if metric == 'pd':
            ax.annotate(f'K=M-1={MAX_CONV}\nConv. limit', xy=(MAX_CONV, 0.12),
                       fontsize=12, color='green', ha='center')
            ax.annotate(f'K={MAX_COP}\nCOP limit\n(ρ(M-1))', xy=(MAX_COP, 0.12),
                       fontsize=12, color='blue', ha='center')

        for name, data in results.items():
            s = ALG_STYLES[name]
            ax.plot(K_values, data[metric], color=s[0], marker=s[1],
                    linestyle=s[2], label=s[3], linewidth=2.5, markersize=10)

        ax.set_xlabel('Number of Sources (K)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Underdetermined DOA: {ylabel} vs Source Count\n'
                     f'M={M} sensors, M_v={M_V}, SNR=15dB, T=256',
                     fontweight='bold')
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xticks(K_values)
        ax.legend(loc=loc, framealpha=0.95)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, fname))
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_snr(snr_values, results):
    for metric, ylabel, yscale, ylim, loc, fname in [
        ('pd', 'Detection Rate (Pd)', 'linear', [-0.05, 1.05], 'lower right', 'fig3_m4_snr_pd.png'),
        ('rmse', 'RMSE (degrees)', 'log', None, 'upper right', 'fig4_m4_snr_rmse.png'),
    ]:
        fig, ax = plt.subplots(figsize=(14, 8))
        for name, data in results.items():
            s = ALG_STYLES[name]
            ax.plot(snr_values, data[metric], color=s[0], marker=s[1],
                    linestyle=s[2], label=s[3])
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'SNR Robustness: {ylabel}\n'
                     f'M={M}, K=4 (Underdetermined: K>M-1={MAX_CONV}), T=128',
                     fontweight='bold')
        ax.set_yscale(yscale)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(loc=loc, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, fname))
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_resolution(spacing_values, results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    for name, data in results.items():
        s = ALG_STYLES[name]
        ax1.plot(spacing_values, data['pd'], color=s[0], marker=s[1],
                 linestyle=s[2], label=s[3])
        ax2.plot(spacing_values, data['rmse'], color=s[0], marker=s[1],
                 linestyle=s[2], label=s[3])
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
    fig.suptitle(f'Close-Spacing Resolution (M={M}, K=3, SNR=15dB, T=512)',
                 fontsize=20, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_m4_resolution.png'))
    plt.close(fig)
    print(f"  Saved fig5_m4_resolution.png")


def plot_snapshots(T_values, results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    for name, data in results.items():
        s = ALG_STYLES[name]
        ax1.semilogx(T_values, data['pd'], color=s[0], marker=s[1],
                     linestyle=s[2], label=s[3], base=2)
        ax2.semilogx(T_values, data['rmse'], color=s[0], marker=s[1],
                     linestyle=s[2], label=s[3], base=2)
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
    fig.suptitle(f'Snapshot Efficiency (M={M}, K=4, SNR=10dB)',
                 fontsize=20, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_m4_snapshots.png'))
    plt.close(fig)
    print(f"  Saved fig6_m4_snapshots.png")


def plot_combined_summary(k_data, snr_data, res_data, snap_data):
    K_values, k_results = k_data
    snr_values, snr_results = snr_data
    spacing_values, res_results = res_data
    T_values, snap_results = snap_data

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) K scaling Pd
    ax = axes[0, 0]
    ax.axvspan(MAX_CONV + 0.5, MAX_COP + 0.5, alpha=0.08, color='blue')
    ax.axvspan(MAX_COP + 0.5, max(K_values) + 0.3, alpha=0.06, color='red')
    ax.axvline(x=MAX_CONV, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=MAX_COP, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    for name, data in k_results.items():
        s = ALG_STYLES[name]
        ax.plot(K_values, data['pd'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3], markersize=5)
    ax.set_xlabel('K (sources)')
    ax.set_ylabel('Pd')
    ax.set_title('(a) K Scaling: Pd')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=5.5, loc='lower left')

    # (0,1) SNR Pd
    ax = axes[0, 1]
    for name, data in snr_results.items():
        s = ALG_STYLES[name]
        ax.plot(snr_values, data['pd'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3], markersize=5)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Pd')
    ax.set_title('(b) SNR: Pd')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=6, loc='lower right')

    # (0,2) Resolution Pd
    ax = axes[0, 2]
    for name, data in res_results.items():
        s = ALG_STYLES[name]
        ax.plot(spacing_values, data['pd'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3], markersize=5)
    ax.set_xlabel('Spacing (deg)')
    ax.set_ylabel('Pd')
    ax.set_title('(c) Resolution: Pd')
    ax.set_ylim([-0.05, 1.05])
    ax.invert_xaxis()
    ax.legend(fontsize=6, loc='lower left')

    # (1,0) K scaling RMSE
    ax = axes[1, 0]
    ax.axvspan(MAX_CONV + 0.5, MAX_COP + 0.5, alpha=0.08, color='blue')
    ax.axvspan(MAX_COP + 0.5, max(K_values) + 0.3, alpha=0.06, color='red')
    ax.axvline(x=MAX_CONV, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=MAX_COP, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    for name, data in k_results.items():
        s = ALG_STYLES[name]
        ax.plot(K_values, data['rmse'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3], markersize=5)
    ax.set_xlabel('K (sources)')
    ax.set_ylabel('RMSE (deg)')
    ax.set_title('(d) K Scaling: RMSE')
    ax.legend(fontsize=5.5, loc='upper left')

    # (1,1) SNR RMSE
    ax = axes[1, 1]
    for name, data in snr_results.items():
        s = ALG_STYLES[name]
        ax.plot(snr_values, data['rmse'], color=s[0], marker=s[1],
                linestyle=s[2], label=s[3], markersize=5)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('RMSE (deg)')
    ax.set_title('(e) SNR: RMSE')
    ax.set_yscale('log')
    ax.legend(fontsize=6, loc='upper right')

    # (1,2) Snapshots RMSE
    ax = axes[1, 2]
    for name, data in snap_results.items():
        s = ALG_STYLES[name]
        ax.semilogx(T_values, data['rmse'], color=s[0], marker=s[1],
                     linestyle=s[2], label=s[3], markersize=5, base=2)
    ax.set_xlabel('Snapshots (T)')
    ax.set_ylabel('RMSE (deg)')
    ax.set_title('(f) Snapshots: RMSE')
    ax.set_xticks(T_values)
    ax.set_xticklabels([str(t) for t in T_values])
    ax.legend(fontsize=6, loc='upper right')

    fig.suptitle(f'COP Family Performance Summary (M={M}, M_v={M_V}, ρ={RHO})\n'
                 f'Conv. limit: K={MAX_CONV} | COP limit: K={MAX_COP}',
                 fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_m4_summary.png'))
    plt.close(fig)
    print(f"  Saved fig_m4_summary.png")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print(f"\n{'#' * 70}")
    print(f"  COP Benchmark: M={M} sensors (Practical Small-Array)")
    print(f"  M_v = {M_V} (virtual), Conv limit = {MAX_CONV}, COP limit = {MAX_COP}")
    print(f"{'#' * 70}")

    # Run experiments
    k_data = exp_k_scaling()
    snr_data = exp_snr()
    res_data = exp_resolution()
    snap_data = exp_snapshots()

    # Generate plots
    print("\n--- Generating M=4 plots ---")
    plot_k_scaling(*k_data)
    plot_snr(*snr_data)
    plot_resolution(*res_data)
    plot_snapshots(*snap_data)
    plot_combined_summary(k_data, snr_data, res_data, snap_data)

    print(f"\n{'=' * 70}")
    print(f"  All M=4 figures saved to: {OUTPUT_DIR}")
    print(f"  Total: 7 figures")
    print(f"{'=' * 70}")
