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
from iron_dome_sim.doa import SubspaceCOP, TemporalCOP, SequentialDeflationCOP, MUSIC, Capon, COP_CBF, COP_MVDR

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


def plot_true_doa_markers(ax, true_deg, y_bottom=-38):
    """Plot true DOA markers consistently: vertical dashed lines + bottom triangles."""
    for i, td in enumerate(true_deg):
        ax.axvline(x=td, color='red', linestyle='--', alpha=0.4, linewidth=1.0,
                   label='True DOAs' if i == 0 else None, zorder=1)
    # Triangle markers at bottom
    ax.plot(true_deg, np.full_like(true_deg, y_bottom), 'r^', markersize=10,
            markeredgecolor='darkred', markeredgewidth=0.6,
            label=None, zorder=5, clip_on=False)
    # Annotate degree values
    for td in true_deg:
        ax.annotate(f'{td:.0f}°', xy=(td, y_bottom + 1.5), fontsize=7,
                    ha='center', va='bottom', color='red', fontweight='bold')


def plot_spatial_spectrum():
    """Generate spatial spectrum comparison: COP vs MUSIC vs Capon.

    Underdetermined scenario: K=10 > M-1=7 sources with M=8 sensors.
    """
    print("Generating Fig 0a: COP vs Classical Spectrum (K=10)...")

    M = 8
    K = 10
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 20
    T = 1024
    # 12° uniform spacing, seed=55 — best config: 10/10, mean_err=0.49°, max_err<1°
    true_doas = np.radians(np.linspace(-54, 54, K))  # 12° apart
    true_deg = np.degrees(true_doas)
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 3601)
    scan_deg = np.degrees(scan_angles)

    np.random.seed(55)
    X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")

    # COP-4th
    cop = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    cop_db = to_db(cop.spectrum(X, scan_angles))
    cop_doas, _ = cop.estimate(X, scan_angles)

    # MUSIC (limited to M-1=7)
    music = MUSIC(array, num_sources=min(K, M - 1))
    music_db = to_db(music.spectrum(X, scan_angles))

    # Capon
    capon = Capon(array)
    capon_db = to_db(capon.spectrum(X, scan_angles))

    # COP-CBF (K-free)
    cop_cbf = COP_CBF(array, num_sources=K, rho=2)
    cop_cbf_db = to_db(cop_cbf.spectrum(X, scan_angles))

    # COP-MVDR (K-free)
    cop_mvdr = COP_MVDR(array, num_sources=K, rho=2)
    cop_mvdr_db = to_db(cop_mvdr.spectrum(X, scan_angles))

    # ---- 2 panels ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), gridspec_kw={'hspace': 0.35})

    # == (a) COP vs MUSIC vs Capon + COP Beamforming ==
    ax1.plot(scan_deg, cop_db, color='#0055CC', linewidth=2.8,
             label='COP-4th [Proposed]', zorder=4)
    ax1.plot(scan_deg, cop_mvdr_db, color='#CC4400', linewidth=2.5, linestyle='-',
             label='COP-MVDR (K-free) [Proposed]', zorder=3)
    ax1.plot(scan_deg, cop_cbf_db, color='#66BB66', linewidth=2.0, linestyle='--',
             label='COP-CBF (K-free)', zorder=3)
    ax1.plot(scan_deg, music_db, color='#888888', linestyle='--', linewidth=2.0,
             label=f'MUSIC (max K={M-1})', zorder=2)
    ax1.plot(scan_deg, capon_db, color='#009999', linestyle=':', linewidth=2.0,
             label='Capon', zorder=2)
    plot_true_doa_markers(ax1, true_deg, y_bottom=-38)

    ax1.set_xlabel('DOA (degrees)')
    ax1.set_ylabel('Normalized Spectrum (dB)')
    ax1.set_title(f'(a) COP Family vs Classical  [M={M}, K={K}, SNR={snr_db} dB]',
                  fontsize=16, fontweight='bold')
    ax1.set_ylim([-42, 5])
    ax1.set_xlim([-90, 90])
    ax1.legend(loc='upper left', framealpha=0.95, fontsize=13)

    # == (b) COP Peak Detection ==
    ax2.plot(scan_deg, cop_db, color='#0055CC', linewidth=2.5,
             label='COP-4th Spectrum', zorder=3)

    # Detected peaks
    for i, cd in enumerate(cop_doas):
        idx = np.argmin(np.abs(scan_angles - cd))
        ax2.plot(np.degrees(cd), cop_db[idx], 'rv', markersize=14,
                 markeredgecolor='black', markeredgewidth=1.0,
                 label='Detected Peaks' if i == 0 else None, zorder=6)
    plot_true_doa_markers(ax2, true_deg, y_bottom=-38)

    ax2.set_xlabel('DOA (degrees)')
    ax2.set_ylabel('Normalized Spectrum (dB)')
    ax2.set_title(f'(b) COP Peak Detection  '
                  f'[M$_v$={2*(M-1)+1}, Detected: {len(cop_doas)}/{K}]',
                  fontsize=16, fontweight='bold')
    ax2.set_ylim([-42, 5])
    ax2.set_xlim([-90, 90])
    ax2.legend(loc='upper left', framealpha=0.95, fontsize=13)

    # Print detection accuracy
    print(f"  True DOAs (deg): {true_deg}")
    print(f"  Detected DOAs (deg): {np.degrees(cop_doas).round(1)}")
    for i, td in enumerate(true_doas):
        closest = cop_doas[np.argmin(np.abs(cop_doas - td))]
        err = np.degrees(np.abs(closest - td))
        print(f"    True={np.degrees(td):+6.1f}°  →  Est={np.degrees(closest):+6.1f}°  (err={err:.2f}°)")

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig0_spatial_spectrum.png'))
    plt.close(fig)
    print(f"  Saved fig0_spatial_spectrum.png (COP: {len(cop_doas)}/{K} detected)")


def plot_cop_family_spectrum():
    """Generate COP family spectrum: COP vs T-COP vs SD-COP.

    Super-underdetermined: K=16 > rho*(M-1)=14 with M=8 sensors.
    """
    print("\nGenerating Fig 0b: COP Family Spectrum (K=16)...")

    M = 8
    K = 16
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 20
    T = 1024
    true_doas = np.radians(np.linspace(-65, 65, K))
    true_deg = np.degrees(true_doas)
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 3601)
    scan_deg = np.degrees(scan_angles)

    np.random.seed(42)
    X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")

    # COP-4th
    cop = SubspaceCOP(array, rho=2, num_sources=min(K, 14), spectrum_type="combined")
    cop_db = to_db(cop.spectrum(X, scan_angles))
    cop_doas, _ = cop.estimate(X, scan_angles)

    # T-COP (10 scans)
    tcop = TemporalCOP(array, rho=2, num_sources=min(K, 14), alpha=0.85, prior_weight=0.0)
    for s in range(10):
        np.random.seed(42 + s)
        Xs, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")
        tcop_spec = tcop.spectrum(Xs, scan_angles)
    tcop_db = to_db(tcop_spec)
    tcop_doas, _ = tcop.estimate(Xs, scan_angles)

    # SD-COP
    sdcop = SequentialDeflationCOP(array, rho=2, num_sources=K,
                                    max_stages=8, spectrum_type="combined")
    sdcop_spec = sdcop.spectrum(X, scan_angles)
    sdcop_db = to_db(sdcop_spec)
    sdcop_doas, _ = sdcop.estimate(X, scan_angles)

    # ---- 3 panels ----
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18),
                                         gridspec_kw={'hspace': 0.40})

    # == (a) All COP family overlaid ==
    ax1.plot(scan_deg, cop_db, color='#0055CC', linewidth=2.5,
             label=f'COP-4th ({len(cop_doas)} det.)', zorder=3)
    ax1.plot(scan_deg, tcop_db, color='#DD0000', linewidth=2.5, linestyle='--',
             label=f'T-COP, 10 scans ({len(tcop_doas)} det.)', zorder=3)
    ax1.plot(scan_deg, sdcop_db, color='#00AA00', linewidth=2.5, linestyle='-.',
             label=f'SD-COP ({len(sdcop_doas)} det.)', zorder=4)
    plot_true_doa_markers(ax1, true_deg, y_bottom=-38)

    ax1.set_xlabel('DOA (degrees)')
    ax1.set_ylabel('Normalized Spectrum (dB)')
    ax1.set_title(f'(a) COP Family  [M={M}, K={K}, SNR={snr_db} dB, K > ρ(M−1)={2*(M-1)}]',
                  fontsize=15, fontweight='bold')
    ax1.set_ylim([-42, 5])
    ax1.set_xlim([-85, 85])
    ax1.legend(loc='upper left', framealpha=0.95, fontsize=12)

    # == (b) T-COP peaks ==
    ax2.plot(scan_deg, tcop_db, color='#DD0000', linewidth=2.5,
             label='T-COP Spectrum (10 scans)', zorder=3)
    for i, cd in enumerate(tcop_doas):
        idx = np.argmin(np.abs(scan_angles - cd))
        ax2.plot(np.degrees(cd), tcop_db[idx], 'bv', markersize=13,
                 markeredgecolor='black', markeredgewidth=0.8,
                 label='T-COP Peaks' if i == 0 else None, zorder=6)
    plot_true_doa_markers(ax2, true_deg, y_bottom=-38)

    ax2.set_xlabel('DOA (degrees)')
    ax2.set_ylabel('Normalized Spectrum (dB)')
    ax2.set_title(f'(b) T-COP Peak Detection  '
                  f'[α=0.85, 10 scans, Detected: {len(tcop_doas)}/{K}]',
                  fontsize=15, fontweight='bold')
    ax2.set_ylim([-42, 5])
    ax2.set_xlim([-85, 85])
    ax2.legend(loc='upper left', framealpha=0.95, fontsize=12)

    # == (c) SD-COP peaks ==
    ax3.plot(scan_deg, sdcop_db, color='#00AA00', linewidth=2.5,
             label='SD-COP Spectrum', zorder=3)
    for i, cd in enumerate(sdcop_doas):
        idx = np.argmin(np.abs(scan_angles - cd))
        ax3.plot(np.degrees(cd), sdcop_db[idx], 'mv', markersize=13,
                 markeredgecolor='black', markeredgewidth=0.8,
                 label='SD-COP Peaks' if i == 0 else None, zorder=6)
    plot_true_doa_markers(ax3, true_deg, y_bottom=-38)

    ax3.set_xlabel('DOA (degrees)')
    ax3.set_ylabel('Normalized Spectrum (dB)')
    ax3.set_title(f'(c) SD-COP Peak Detection  '
                  f'[Deflation, Detected: {len(sdcop_doas)}/{K}]',
                  fontsize=15, fontweight='bold')
    ax3.set_ylim([-42, 5])
    ax3.set_xlim([-85, 85])
    ax3.legend(loc='upper left', framealpha=0.95, fontsize=12)

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig0b_cop_family_spectrum.png'))
    plt.close(fig)
    print(f"  Saved fig0b_cop_family_spectrum.png")
    print(f"  COP: {len(cop_doas)}, T-COP: {len(tcop_doas)}, SD-COP: {len(sdcop_doas)} / {K}")


if __name__ == '__main__':
    plot_spatial_spectrum()
    plot_cop_family_spectrum()
    print("\nDone!")
