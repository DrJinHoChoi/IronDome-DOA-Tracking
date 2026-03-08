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
    'axes.titlesize': 20,
    'legend.fontsize': 13,
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
    """Generate spatial spectrum comparison: COP vs MUSIC vs Capon."""
    print("Generating Fig 0a: COP vs Classical Spectrum...")

    M = 8
    K = 10  # Underdetermined: K > M-1 = 7
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 256
    true_doas = np.radians([-45, -30, -15, -5, 5, 15, 25, 35, 50, 60])
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    scan_deg = np.degrees(scan_angles)

    np.random.seed(42)
    X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")

    # COP-4th
    cop = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    cop_db = to_db(cop.spectrum(X, scan_angles))
    cop_doas, _ = cop.estimate(X, scan_angles)

    # MUSIC
    music = MUSIC(array, num_sources=min(K, M - 1))
    music_db = to_db(music.spectrum(X, scan_angles))

    # Capon
    capon = Capon(array)
    capon_db = to_db(capon.spectrum(X, scan_angles))

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Top: All three spectra
    ax1.plot(scan_deg, cop_db, '#0055CC', linewidth=2.5, label='COP-4th [Proposed]', zorder=3)
    ax1.plot(scan_deg, music_db, '#AAAAAA', linestyle='--', linewidth=2.0, label='MUSIC', zorder=2)
    ax1.plot(scan_deg, capon_db, '#008888', linestyle=':', linewidth=2.0, label='Capon', zorder=2)

    for i, td in enumerate(true_doas):
        ax1.axvline(x=np.degrees(td), color='red', linestyle='-', alpha=0.3, linewidth=1.0,
                   label='True DOAs' if i == 0 else None)

    ax1.set_xlabel('DOA (degrees)')
    ax1.set_ylabel('Normalized Spectrum (dB)')
    ax1.set_title(f'(a) Spatial Spectrum: COP vs Classical Methods\n'
                  f'M={M}, K={K} (underdetermined: K > M\u22121={M-1}), SNR={snr_db}dB',
                  fontweight='bold')
    ax1.set_ylim([-40, 3])
    ax1.set_xlim([-90, 90])
    ax1.legend(loc='lower center', ncol=4, framealpha=0.9)

    # Bottom: COP with peak detection
    ax2.plot(scan_deg, cop_db, '#0055CC', linewidth=2.5, label='COP-4th spectrum')
    for i, cd in enumerate(cop_doas):
        idx = np.argmin(np.abs(scan_angles - cd))
        ax2.plot(np.degrees(cd), cop_db[idx], 'rv', markersize=12,
                markeredgecolor='black', markeredgewidth=0.8,
                label='Detected peaks' if i == 0 else None, zorder=5)
    for i, td in enumerate(true_doas):
        ax2.axvline(x=np.degrees(td), color='red', linestyle='-', alpha=0.3, linewidth=1.0,
                   label='True DOAs' if i == 0 else None)

    ax2.set_xlabel('DOA (degrees)')
    ax2.set_ylabel('Normalized Spectrum (dB)')
    ax2.set_title(f'(b) COP Peak Detection: M_v=\u03c1(M\u22121)+1={2*(M-1)+1}, '
                  f'Detected {len(cop_doas)}/{K} sources', fontweight='bold')
    ax2.set_ylim([-40, 3])
    ax2.set_xlim([-90, 90])
    ax2.legend(loc='lower center', ncol=3, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig0_spatial_spectrum.png'))
    plt.close(fig)
    print(f"  Saved fig0_spatial_spectrum.png (COP: {len(cop_doas)}/{K} detected)")


def plot_cop_family_spectrum():
    """Generate COP family spectrum: COP vs T-COP vs SD-COP."""
    print("Generating Fig 0b: COP Family Spectrum...")

    M = 8
    K = 16  # Super-underdetermined: K > rho*(M-1) = 14
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 512
    true_doas = np.radians(np.linspace(-60, 60, K))
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    scan_deg = np.degrees(scan_angles)

    np.random.seed(42)
    X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")

    # COP-4th
    cop = SubspaceCOP(array, rho=2, num_sources=min(K, 14), spectrum_type="combined")
    cop_db = to_db(cop.spectrum(X, scan_angles))
    cop_doas, _ = cop.estimate(X, scan_angles)

    # T-COP (multi-scan accumulation)
    tcop = TemporalCOP(array, rho=2, num_sources=min(K, 14), alpha=0.85, prior_weight=0.0)
    # Simulate 5 scans
    for s in range(5):
        np.random.seed(42 + s)
        Xs, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")
        tcop_spec = tcop.spectrum(Xs, scan_angles)
    tcop_db = to_db(tcop_spec)
    tcop_doas, _ = tcop.estimate(Xs, scan_angles)

    # SD-COP
    sdcop = SequentialDeflationCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    sdcop_spec = sdcop.spectrum(X, scan_angles)
    sdcop_db = to_db(sdcop_spec)
    sdcop_doas, _ = sdcop.estimate(X, scan_angles)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Top: All COP family spectra
    ax1.plot(scan_deg, cop_db, '#0055CC', linewidth=2.5, label=f'COP-4th ({len(cop_doas)} det.)', zorder=2)
    ax1.plot(scan_deg, tcop_db, '#DD0000', linewidth=2.5, linestyle='--',
             label=f'T-COP, 5 scans ({len(tcop_doas)} det.)', zorder=2)
    ax1.plot(scan_deg, sdcop_db, '#00AA00', linewidth=2.5, linestyle='-.',
             label=f'SD-COP ({len(sdcop_doas)} det.)', zorder=3)

    for i, td in enumerate(true_doas):
        ax1.axvline(x=np.degrees(td), color='red', linestyle='-', alpha=0.25, linewidth=0.8,
                   label='True DOAs' if i == 0 else None)

    ax1.set_xlabel('DOA (degrees)')
    ax1.set_ylabel('Normalized Spectrum (dB)')
    ax1.set_title(f'(a) COP Family Spectra: Super-Underdetermined\n'
                  f'M={M}, K={K} (K > \u03c1(M\u22121)={2*(M-1)}), SNR={snr_db}dB',
                  fontweight='bold')
    ax1.set_ylim([-40, 3])
    ax1.set_xlim([-90, 90])
    ax1.legend(loc='lower center', ncol=4, framealpha=0.9, fontsize=12)

    # Bottom: SD-COP with detected peaks
    ax2.plot(scan_deg, sdcop_db, '#00AA00', linewidth=2.5, label='SD-COP spectrum')
    for i, cd in enumerate(sdcop_doas):
        idx = np.argmin(np.abs(scan_angles - cd))
        ax2.plot(np.degrees(cd), sdcop_db[idx], 'rv', markersize=12,
                markeredgecolor='black', markeredgewidth=0.8,
                label='SD-COP peaks' if i == 0 else None, zorder=5)
    for i, td in enumerate(true_doas):
        ax2.axvline(x=np.degrees(td), color='red', linestyle='-', alpha=0.25, linewidth=0.8,
                   label='True DOAs' if i == 0 else None)

    ax2.set_xlabel('DOA (degrees)')
    ax2.set_ylabel('Normalized Spectrum (dB)')
    ax2.set_title(f'(b) SD-COP Peak Detection: {len(sdcop_doas)}/{K} sources detected\n'
                  f'Sequential deflation extends capacity beyond \u03c1(M\u22121)={2*(M-1)}',
                  fontweight='bold')
    ax2.set_ylim([-40, 3])
    ax2.set_xlim([-90, 90])
    ax2.legend(loc='lower center', ncol=3, framealpha=0.9, fontsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig0b_cop_family_spectrum.png'))
    plt.close(fig)
    print(f"  Saved fig0b_cop_family_spectrum.png")
    print(f"  COP: {len(cop_doas)}, T-COP: {len(tcop_doas)}, SD-COP: {len(sdcop_doas)} / {K}")


if __name__ == '__main__':
    plot_spatial_spectrum()
    plot_cop_family_spectrum()
    print("Done!")
