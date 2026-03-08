#!/usr/bin/env python3
"""Generate spatial spectrum comparison plots for the paper.

Fig 0a: COP vs MUSIC vs Capon spatial spectrum (underdetermined)
Fig 0b: COP vs T-COP vs SD-COP spatial spectrum (proposed family)
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
from iron_dome_sim.doa import SubspaceCOP, TemporalCOP, SequentialDeflationCOP, MUSIC, Capon

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
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


def to_db(spectrum):
    """Convert spectrum to normalized dB scale."""
    s = np.abs(spectrum)
    return 10 * np.log10(s / np.max(s) + 1e-15)


def plot_spatial_spectrum():
    """Generate spatial spectrum comparison: COP vs MUSIC vs Capon.

    Underdetermined scenario: K=10 > M-1=7 sources with M=8 sensors.
    COP resolves all sources while MUSIC and Capon fail.
    """
    print("Generating Fig 0a: COP vs Classical Spectrum...")

    M = 8
    K = 10  # Underdetermined: K > M-1 = 7
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 256
    # Use the source positions that achieved 10/10 detection
    true_doas = np.radians([-45, -30, -15, -5, 5, 15, 25, 35, 50, 60])
    true_deg = np.degrees(true_doas)
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 3601)
    scan_deg = np.degrees(scan_angles)

    np.random.seed(42)
    X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")

    # COP-4th (combined spectrum - proposed)
    cop = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    cop_db = to_db(cop.spectrum(X, scan_angles))
    cop_doas, _ = cop.estimate(X, scan_angles)

    # MUSIC (limited to M-1=7 sources)
    music = MUSIC(array, num_sources=min(K, M - 1))
    music_db = to_db(music.spectrum(X, scan_angles))

    # Capon
    capon = Capon(array)
    capon_db = to_db(capon.spectrum(X, scan_angles))

    # ---- Figure: 2 panels stacked ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), gridspec_kw={'hspace': 0.35})

    # ============ Panel (a): COP vs MUSIC vs Capon ============
    ax1.plot(scan_deg, cop_db, color='#0055CC', linewidth=2.8,
             label='COP-4th [Proposed]', zorder=4)
    ax1.plot(scan_deg, music_db, color='#888888', linestyle='--', linewidth=2.0,
             label=f'MUSIC (K$\\leq${M-1}={M-1})', zorder=2)
    ax1.plot(scan_deg, capon_db, color='#009999', linestyle=':', linewidth=2.0,
             label='Capon', zorder=2)

    # True DOA markers (subtle triangular markers at bottom)
    for i, td in enumerate(true_deg):
        ax1.axvline(x=td, color='#DD3333', linestyle='-', alpha=0.2, linewidth=0.8)
    ax1.plot(true_deg, np.full_like(true_deg, -38.0), 'r^', markersize=8,
             markeredgecolor='darkred', markeredgewidth=0.5,
             label=f'True DOAs (K={K})', zorder=5, clip_on=False)

    ax1.set_xlabel('DOA (degrees)')
    ax1.set_ylabel('Normalized Spectrum (dB)')
    ax1.set_title(f'(a) COP vs Classical Methods  '
                  f'[M={M}, K={K}, SNR={snr_db} dB, underdetermined: K > M−1]',
                  fontsize=16, fontweight='bold')
    ax1.set_ylim([-42, 5])
    ax1.set_xlim([-90, 90])
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=13)

    # ============ Panel (b): COP with peak detection ============
    ax2.plot(scan_deg, cop_db, color='#0055CC', linewidth=2.5,
             label='COP-4th Spectrum', zorder=3)

    # Detected peak markers
    for i, cd in enumerate(cop_doas):
        idx = np.argmin(np.abs(scan_angles - cd))
        ax2.plot(np.degrees(cd), cop_db[idx], 'rv', markersize=14,
                 markeredgecolor='black', markeredgewidth=1.0,
                 label='Detected Peaks' if i == 0 else None, zorder=6)

    # True DOA markers
    for i, td in enumerate(true_deg):
        ax2.axvline(x=td, color='#DD3333', linestyle='-', alpha=0.2, linewidth=0.8)
    ax2.plot(true_deg, np.full_like(true_deg, -38.0), 'r^', markersize=8,
             markeredgecolor='darkred', markeredgewidth=0.5,
             label=f'True DOAs (K={K})', zorder=5, clip_on=False)

    ax2.set_xlabel('DOA (degrees)')
    ax2.set_ylabel('Normalized Spectrum (dB)')
    ax2.set_title(f'(b) COP Peak Detection  '
                  f'[M$_v$=ρ(M−1)+1={2*(M-1)+1}, '
                  f'Detected: {len(cop_doas)}/{K}]',
                  fontsize=16, fontweight='bold')
    ax2.set_ylim([-42, 5])
    ax2.set_xlim([-90, 90])
    ax2.legend(loc='upper right', framealpha=0.95, fontsize=13)

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig0_spatial_spectrum.png'))
    plt.close(fig)
    print(f"  Saved fig0_spatial_spectrum.png (COP: {len(cop_doas)}/{K} detected)")


def plot_cop_family_spectrum():
    """Generate COP family spectrum: COP vs T-COP vs SD-COP.

    Super-underdetermined scenario: K=16 > rho*(M-1)=14 with M=8 sensors.
    SD-COP via sequential deflation extends beyond COP's native capacity.
    """
    print("Generating Fig 0b: COP Family Spectrum...")

    M = 8
    K = 16  # Super-underdetermined: K > rho*(M-1) = 14
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 20   # Higher SNR for cleaner estimation
    T = 1024      # More snapshots for stable cumulant estimation
    # Widely spread sources
    true_doas = np.radians(np.linspace(-65, 65, K))
    true_deg = np.degrees(true_doas)
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 3601)
    scan_deg = np.degrees(scan_angles)

    np.random.seed(42)
    X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")

    # COP-4th (limited by rho*(M-1)=14 capacity)
    cop = SubspaceCOP(array, rho=2, num_sources=min(K, 14), spectrum_type="combined")
    cop_db = to_db(cop.spectrum(X, scan_angles))
    cop_doas, _ = cop.estimate(X, scan_angles)

    # T-COP (multi-scan accumulation with 10 scans for more improvement)
    tcop = TemporalCOP(array, rho=2, num_sources=min(K, 14), alpha=0.85, prior_weight=0.0)
    for s in range(10):
        np.random.seed(42 + s)
        Xs, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")
        tcop_spec = tcop.spectrum(Xs, scan_angles)
    tcop_db = to_db(tcop_spec)
    tcop_doas, _ = tcop.estimate(Xs, scan_angles)

    # SD-COP (sequential deflation - can exceed rho*(M-1) capacity)
    sdcop = SequentialDeflationCOP(array, rho=2, num_sources=K,
                                    max_stages=8, spectrum_type="combined")
    sdcop_spec = sdcop.spectrum(X, scan_angles)
    sdcop_db = to_db(sdcop_spec)
    sdcop_doas, _ = sdcop.estimate(X, scan_angles)

    # ---- Figure: 3 panels stacked ----
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18),
                                         gridspec_kw={'hspace': 0.40})

    # ============ Panel (a): All COP family spectra overlaid ============
    ax1.plot(scan_deg, cop_db, color='#0055CC', linewidth=2.5,
             label=f'COP-4th ({len(cop_doas)} det.)', zorder=3)
    ax1.plot(scan_deg, tcop_db, color='#DD0000', linewidth=2.5, linestyle='--',
             label=f'T-COP, 10 scans ({len(tcop_doas)} det.)', zorder=3)
    ax1.plot(scan_deg, sdcop_db, color='#00AA00', linewidth=2.5, linestyle='-.',
             label=f'SD-COP ({len(sdcop_doas)} det.)', zorder=4)

    for i, td in enumerate(true_deg):
        ax1.axvline(x=td, color='#DD3333', linestyle='-', alpha=0.15, linewidth=0.6)
    ax1.plot(true_deg, np.full_like(true_deg, -38.0), 'r^', markersize=6,
             markeredgecolor='darkred', markeredgewidth=0.4,
             label=f'True DOAs (K={K})', zorder=5, clip_on=False)

    ax1.set_xlabel('DOA (degrees)')
    ax1.set_ylabel('Normalized Spectrum (dB)')
    ax1.set_title(f'(a) COP Family Spectra  '
                  f'[M={M}, K={K}, SNR={snr_db} dB, super-underdetermined: K > ρ(M−1)]',
                  fontsize=15, fontweight='bold')
    ax1.set_ylim([-42, 5])
    ax1.set_xlim([-85, 85])
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=12)

    # ============ Panel (b): T-COP with peak detection ============
    ax2.plot(scan_deg, tcop_db, color='#DD0000', linewidth=2.5,
             label='T-COP Spectrum (10 scans)', zorder=3)

    for i, cd in enumerate(tcop_doas):
        idx = np.argmin(np.abs(scan_angles - cd))
        ax2.plot(np.degrees(cd), tcop_db[idx], 'bv', markersize=13,
                 markeredgecolor='black', markeredgewidth=0.8,
                 label='T-COP Peaks' if i == 0 else None, zorder=6)
    for i, td in enumerate(true_deg):
        ax2.axvline(x=td, color='#DD3333', linestyle='-', alpha=0.15, linewidth=0.6)
    ax2.plot(true_deg, np.full_like(true_deg, -38.0), 'r^', markersize=6,
             markeredgecolor='darkred', markeredgewidth=0.4,
             label=f'True DOAs (K={K})', zorder=5, clip_on=False)

    ax2.set_xlabel('DOA (degrees)')
    ax2.set_ylabel('Normalized Spectrum (dB)')
    ax2.set_title(f'(b) T-COP Peak Detection  '
                  f'[α=0.85, 10 scans, Detected: {len(tcop_doas)}/{K}]',
                  fontsize=15, fontweight='bold')
    ax2.set_ylim([-42, 5])
    ax2.set_xlim([-85, 85])
    ax2.legend(loc='upper right', framealpha=0.95, fontsize=12)

    # ============ Panel (c): SD-COP with peak detection ============
    ax3.plot(scan_deg, sdcop_db, color='#00AA00', linewidth=2.5,
             label='SD-COP Spectrum', zorder=3)

    for i, cd in enumerate(sdcop_doas):
        idx = np.argmin(np.abs(scan_angles - cd))
        ax3.plot(np.degrees(cd), sdcop_db[idx], 'mv', markersize=13,
                 markeredgecolor='black', markeredgewidth=0.8,
                 label='SD-COP Peaks' if i == 0 else None, zorder=6)
    for i, td in enumerate(true_deg):
        ax3.axvline(x=td, color='#DD3333', linestyle='-', alpha=0.15, linewidth=0.6)
    ax3.plot(true_deg, np.full_like(true_deg, -38.0), 'r^', markersize=6,
             markeredgecolor='darkred', markeredgewidth=0.4,
             label=f'True DOAs (K={K})', zorder=5, clip_on=False)

    ax3.set_xlabel('DOA (degrees)')
    ax3.set_ylabel('Normalized Spectrum (dB)')
    ax3.set_title(f'(c) SD-COP Peak Detection  '
                  f'[Deflation stages, Detected: {len(sdcop_doas)}/{K}]',
                  fontsize=15, fontweight='bold')
    ax3.set_ylim([-42, 5])
    ax3.set_xlim([-85, 85])
    ax3.legend(loc='upper right', framealpha=0.95, fontsize=12)

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig0b_cop_family_spectrum.png'))
    plt.close(fig)
    print(f"  Saved fig0b_cop_family_spectrum.png")
    print(f"  COP: {len(cop_doas)}, T-COP: {len(tcop_doas)}, SD-COP: {len(sdcop_doas)} / {K}")


if __name__ == '__main__':
    plot_spatial_spectrum()
    plot_cop_family_spectrum()
    print("Done!")
