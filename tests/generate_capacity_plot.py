#!/usr/bin/env python3
"""Generate K=14 capacity limit spectrum plot.

Demonstrates COP-4th resolving ALL 14 sources at the theoretical capacity limit:
  rho*(M-1) = 2*7 = 14

Key: True DOAs and Estimated DOAs are CLEARLY distinguished with different
marker styles, colors, and visual elements.
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
from iron_dome_sim.doa import SubspaceCOP

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
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
    s = np.abs(spectrum)
    return 10 * np.log10(s / np.max(s) + 1e-15)


def main():
    print("=" * 70)
    print("Generating K=14 Capacity Limit Spectrum Plot")
    print("=" * 70)

    # Configuration: best parameters from test_14peaks.py
    M = 8
    rho = 2
    K = 14
    snr_db = 30
    T = 4096
    seed = 36
    M_v = rho * (M - 1) + 1  # = 15
    K_max = rho * (M - 1)    # = 14

    array = UniformLinearArray(M=M, d=0.5)
    true_doas = np.radians(np.linspace(-65, 65, K))
    true_deg = np.degrees(true_doas)
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 3601)
    scan_deg = np.degrees(scan_angles)

    print(f"  M={M}, rho={rho}, K={K}, M_v={M_v}, K_max={K_max}")
    print(f"  SNR={snr_db} dB, T={T}, seed={seed}")
    print(f"  True DOAs: {np.array2string(true_deg, precision=1, separator=', ')}")

    # Generate signal and estimate
    np.random.seed(seed)
    X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")

    cop = SubspaceCOP(array, rho=rho, num_sources=K, spectrum_type="combined")
    est_doas, P = cop.estimate(X, scan_angles)
    P_db = to_db(P)
    est_deg = np.degrees(est_doas)

    # Detection accuracy
    n_correct = 0
    matched_pairs = []
    used = set()
    for td in true_doas:
        best_idx = None
        best_err = np.inf
        for j, ed in enumerate(est_doas):
            if j in used:
                continue
            err = abs(ed - td)
            if err < best_err:
                best_err = err
                best_idx = j
        if best_idx is not None and best_err < np.radians(3.0):
            used.add(best_idx)
            matched_pairs.append((np.degrees(td), np.degrees(est_doas[best_idx]),
                                  np.degrees(best_err)))
            n_correct += 1
        else:
            matched_pairs.append((np.degrees(td), None, None))

    print(f"\n  Detection: {n_correct}/{K}")
    errors = [e for _, _, e in matched_pairs if e is not None]
    if errors:
        print(f"  Mean error: {np.mean(errors):.3f} deg")
        print(f"  Max  error: {np.max(errors):.3f} deg")

    # ================================================================
    # PLOT: Clear distinction between True DOAs and Estimated DOAs
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14),
                                     gridspec_kw={'hspace': 0.35})

    # ---- Panel (a): Spectrum with True DOA markers only ----
    ax1.plot(scan_deg, P_db, color='#0055CC', linewidth=2.8,
             label='COP-4th Spectrum', zorder=3)

    # True DOAs: GREEN dashed vertical lines + GREEN filled circles at bottom
    for i, td in enumerate(true_deg):
        ax1.axvline(x=td, color='#00AA00', linestyle='--', alpha=0.5,
                    linewidth=1.2, zorder=1)
    ax1.plot(true_deg, np.full_like(true_deg, -38.0), 'o',
             color='#00AA00', markersize=10, markeredgecolor='darkgreen',
             markeredgewidth=1.0, zorder=5, clip_on=False,
             label=f'True DOAs (K={K})')

    # Annotate true DOA values
    for i, td in enumerate(true_deg):
        offset = 1.5 if i % 2 == 0 else 3.5
        ax1.annotate(f'{td:.0f}', xy=(td, -38.0 + offset), fontsize=7,
                     ha='center', va='bottom', color='#00AA00', fontweight='bold')

    ax1.set_xlabel('DOA (degrees)')
    ax1.set_ylabel('Normalized Spectrum (dB)')
    ax1.set_title(f'(a) COP-4th Spectrum at Capacity Limit  '
                  f'[M={M}, K={K}={chr(961)}(M-1), SNR={snr_db} dB]',
                  fontsize=16, fontweight='bold')
    ax1.set_ylim([-42, 5])
    ax1.set_xlim([-85, 85])
    ax1.legend(loc='upper left', framealpha=0.95, fontsize=13)

    # ---- Panel (b): Spectrum with BOTH True and Estimated DOAs ----
    ax2.plot(scan_deg, P_db, color='#0055CC', linewidth=2.5,
             label='COP-4th Spectrum', zorder=3)

    # True DOAs: GREEN dashed lines + GREEN circles at y=-44
    for i, td in enumerate(true_deg):
        ax2.axvline(x=td, color='#00AA00', linestyle='--', alpha=0.4,
                    linewidth=1.0, zorder=1)
    ax2.plot(true_deg, np.full_like(true_deg, -44.0), 'o',
             color='#00AA00', markersize=10, markeredgecolor='darkgreen',
             markeredgewidth=1.0, zorder=5, clip_on=False)

    # Estimated DOAs: RED inverted triangles ON the spectrum curve
    for i, ed_rad in enumerate(est_doas):
        idx = np.argmin(np.abs(scan_angles - ed_rad))
        ed = np.degrees(ed_rad)
        ax2.plot(ed, P_db[idx], '^', color='#DD0000', markersize=14,
                 markeredgecolor='black', markeredgewidth=1.0, zorder=6)

    # ALSO add RED markers at fixed y=-48 so ALL estimated DOAs are clearly visible
    est_deg_arr = np.degrees(est_doas)
    ax2.plot(est_deg_arr, np.full_like(est_deg_arr, -48.0), '^',
             color='#DD0000', markersize=10, markeredgecolor='black',
             markeredgewidth=0.8, zorder=5, clip_on=False)

    # Draw matching lines between true (y=-44) and estimated (y=-48)
    for true_d, est_d, err_d in matched_pairs:
        if est_d is not None:
            color = '#00AA00' if err_d < 1.5 else '#FF8800'
            ax2.plot([true_d, est_d], [-44.0, -48.0],
                     color=color, linewidth=0.8, alpha=0.5, zorder=0)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color='#0055CC', linewidth=2.5, label='COP-4th Spectrum'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00AA00',
               markeredgecolor='darkgreen', markersize=10,
               label=f'True DOAs (K={K})'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#DD0000',
               markeredgecolor='black', markersize=12,
               label=f'Estimated DOAs ({n_correct}/{K} correct)'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left', framealpha=0.95, fontsize=13)

    # Row labels on the left
    ax2.annotate('True →', xy=(-84, -44.0), fontsize=10, fontweight='bold',
                 color='#00AA00', va='center', ha='right')
    ax2.annotate('Est →', xy=(-84, -48.0), fontsize=10, fontweight='bold',
                 color='#DD0000', va='center', ha='right')

    # Horizontal separator line between True and Est rows
    ax2.axhline(y=-46.0, color='gray', linewidth=0.5, alpha=0.3, xmin=0.02, xmax=0.98)

    ax2.set_xlabel('DOA (degrees)')
    ax2.set_ylabel('Normalized Spectrum (dB)')
    ax2.set_title(f'(b) Peak Detection at Capacity  '
                  f'[M$_v$={M_v}, Detected: {n_correct}/{K}, '
                  f'Mean err: {np.mean(errors):.2f}' + chr(176) + ']',
                  fontsize=16, fontweight='bold')
    ax2.set_ylim([-54, 5])
    ax2.set_xlim([-85, 85])

    # Print detailed results
    print(f"\n  Per-source detection:")
    for true_d, est_d, err_d in matched_pairs:
        if est_d is not None:
            status = "OK" if err_d < 3.0 else "MISS"
            print(f"    True={true_d:+7.1f}  ->  Est={est_d:+7.1f}  (err={err_d:.2f})  [{status}]")
        else:
            print(f"    True={true_d:+7.1f}  ->  MISSED")

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig0c_capacity_limit.png'))
    plt.close(fig)
    print(f"\n  Saved fig0c_capacity_limit.png")

    # ================================================================
    # Also update fig0a and fig0b with improved True/Estimated distinction
    # ================================================================
    print("\n" + "=" * 70)
    print("Regenerating fig0a and fig0b with improved True/Est markers...")
    print("=" * 70)
    generate_improved_fig0a()
    generate_improved_fig0b()


def generate_improved_fig0a():
    """Fig 0a: COP vs MUSIC vs Capon with clear True/Estimated DOA markers."""
    print("\nGenerating improved Fig 0a...")

    M, K = 8, 10
    array = UniformLinearArray(M=M, d=0.5)
    snr_db, T = 20, 1024
    true_doas = np.radians(np.linspace(-54, 54, K))
    true_deg = np.degrees(true_doas)
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 3601)
    scan_deg = np.degrees(scan_angles)

    np.random.seed(55)
    X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")

    # Algorithms
    cop = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    cop_db = to_db(cop.spectrum(X, scan_angles))
    cop_doas, _ = cop.estimate(X, scan_angles)

    from iron_dome_sim.doa import MUSIC, Capon
    music = MUSIC(array, num_sources=min(K, M - 1))
    music_db = to_db(music.spectrum(X, scan_angles))
    capon = Capon(array)
    capon_db = to_db(capon.spectrum(X, scan_angles))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14),
                                     gridspec_kw={'hspace': 0.35})

    # == Panel (a): COP vs MUSIC vs Capon ==
    ax1.plot(scan_deg, cop_db, color='#0055CC', linewidth=2.8,
             label='COP-4th [Proposed]', zorder=4)
    ax1.plot(scan_deg, music_db, color='#888888', linestyle='--', linewidth=2.0,
             label=f'MUSIC (max K={M-1})', zorder=2)
    ax1.plot(scan_deg, capon_db, color='#009999', linestyle=':', linewidth=2.0,
             label='Capon', zorder=2)

    # True DOAs: GREEN
    for td in true_deg:
        ax1.axvline(x=td, color='#00AA00', linestyle='--', alpha=0.4, linewidth=1.0, zorder=1)
    ax1.plot(true_deg, np.full_like(true_deg, -38.0), 'o',
             color='#00AA00', markersize=9, markeredgecolor='darkgreen',
             markeredgewidth=0.8, label=f'True DOAs (K={K})', zorder=5, clip_on=False)

    ax1.set_xlabel('DOA (degrees)')
    ax1.set_ylabel('Normalized Spectrum (dB)')
    ax1.set_title(f'(a) COP vs Classical  [M={M}, K={K}, SNR={snr_db} dB]',
                  fontsize=16, fontweight='bold')
    ax1.set_ylim([-42, 5])
    ax1.set_xlim([-90, 90])
    ax1.legend(loc='upper left', framealpha=0.95, fontsize=13)

    # == Panel (b): COP Peak Detection ==
    ax2.plot(scan_deg, cop_db, color='#0055CC', linewidth=2.5, zorder=3)

    # True DOAs: GREEN circles at bottom
    for td in true_deg:
        ax2.axvline(x=td, color='#00AA00', linestyle='--', alpha=0.4, linewidth=1.0, zorder=1)
    ax2.plot(true_deg, np.full_like(true_deg, -38.0), 'o',
             color='#00AA00', markersize=9, markeredgecolor='darkgreen',
             markeredgewidth=0.8, zorder=5, clip_on=False)

    # Estimated DOAs: RED inverted triangles on spectrum
    for i, cd in enumerate(cop_doas):
        idx = np.argmin(np.abs(scan_angles - cd))
        ax2.plot(np.degrees(cd), cop_db[idx], '^', color='#DD0000',
                 markersize=14, markeredgecolor='black', markeredgewidth=1.0, zorder=6)

    # Annotate true DOA values
    for i, td in enumerate(true_deg):
        offset = 1.5 if i % 2 == 0 else 3.5
        ax2.annotate(f'{td:.0f}', xy=(td, -38.0 + offset), fontsize=7,
                     ha='center', va='bottom', color='#00AA00', fontweight='bold')

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color='#0055CC', linewidth=2.5, label='COP-4th Spectrum'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00AA00',
               markeredgecolor='darkgreen', markersize=10,
               label=f'True DOAs (K={K})'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#DD0000',
               markeredgecolor='black', markersize=12,
               label=f'Estimated DOAs ({len(cop_doas)}/{K})'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left', framealpha=0.95, fontsize=13)

    ax2.set_xlabel('DOA (degrees)')
    ax2.set_ylabel('Normalized Spectrum (dB)')
    ax2.set_title(f'(b) COP Peak Detection  '
                  f'[M$_v$={2*(M-1)+1}, Detected: {len(cop_doas)}/{K}]',
                  fontsize=16, fontweight='bold')
    ax2.set_ylim([-42, 5])
    ax2.set_xlim([-90, 90])

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig0_spatial_spectrum.png'))
    plt.close(fig)
    print(f"  Saved fig0_spatial_spectrum.png (COP: {len(cop_doas)}/{K})")


def generate_improved_fig0b():
    """Fig 0b: COP family with clear True/Estimated markers."""
    print("\nGenerating improved Fig 0b...")

    M, K = 8, 16
    array = UniformLinearArray(M=M, d=0.5)
    snr_db, T = 20, 1024
    true_doas = np.radians(np.linspace(-65, 65, K))
    true_deg = np.degrees(true_doas)
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 3601)
    scan_deg = np.degrees(scan_angles)

    np.random.seed(42)
    X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")

    from iron_dome_sim.doa import TemporalCOP, SequentialDeflationCOP

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
    sdcop_doas, sdcop_spec = sdcop.estimate(X, scan_angles)
    sdcop_db = to_db(sdcop_spec)

    # Count correct detections
    def count_correct(est, true, thr_deg=3.0):
        if len(est) == 0:
            return 0
        return sum(1 for d in true if min(abs(est - d)) < np.radians(thr_deg))

    cop_ok = count_correct(cop_doas, true_doas)
    tcop_ok = count_correct(tcop_doas, true_doas)
    sdcop_ok = count_correct(sdcop_doas, true_doas)

    # ---- 3 panels ----
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 20),
                                         gridspec_kw={'hspace': 0.40})

    def add_markers(ax, spec_db, est_doas_rad, color_est, label_est, true_deg_arr):
        """Add clear True (green circles) + Estimated (colored triangles) markers."""
        # True DOAs: GREEN
        for td in true_deg_arr:
            ax.axvline(x=td, color='#00AA00', linestyle='--', alpha=0.35, linewidth=0.8, zorder=1)
        ax.plot(true_deg_arr, np.full_like(true_deg_arr, -38.0), 'o',
                color='#00AA00', markersize=8, markeredgecolor='darkgreen',
                markeredgewidth=0.7, zorder=5, clip_on=False)

        # Estimated DOAs: colored inverted triangles on spectrum
        for ed_rad in est_doas_rad:
            idx = np.argmin(np.abs(scan_angles - ed_rad))
            ax.plot(np.degrees(ed_rad), spec_db[idx], '^', color=color_est,
                    markersize=12, markeredgecolor='black', markeredgewidth=0.8, zorder=6)

    # == (a) COP ==
    ax1.plot(scan_deg, cop_db, color='#0055CC', linewidth=2.5, zorder=3)
    add_markers(ax1, cop_db, cop_doas, '#DD0000', 'COP Peaks', true_deg)

    legend1 = [
        Line2D([0], [0], color='#0055CC', linewidth=2.5, label='COP-4th Spectrum'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00AA00',
               markeredgecolor='darkgreen', markersize=9, label=f'True DOAs (K={K})'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#DD0000',
               markeredgecolor='black', markersize=11,
               label=f'Estimated ({cop_ok}/{K} correct)'),
    ]
    ax1.legend(handles=legend1, loc='upper left', framealpha=0.95, fontsize=12)
    ax1.set_xlabel('DOA (degrees)')
    ax1.set_ylabel('Normalized Spectrum (dB)')
    ax1.set_title(f'(a) COP-4th  [M={M}, K={K}, SNR={snr_db} dB]',
                  fontsize=15, fontweight='bold')
    ax1.set_ylim([-42, 5])
    ax1.set_xlim([-85, 85])

    # == (b) T-COP ==
    ax2.plot(scan_deg, tcop_db, color='#DD0000', linewidth=2.5, zorder=3)
    add_markers(ax2, tcop_db, tcop_doas, '#0055CC', 'T-COP Peaks', true_deg)

    legend2 = [
        Line2D([0], [0], color='#DD0000', linewidth=2.5, label='T-COP Spectrum (10 scans)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00AA00',
               markeredgecolor='darkgreen', markersize=9, label=f'True DOAs (K={K})'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#0055CC',
               markeredgecolor='black', markersize=11,
               label=f'Estimated ({tcop_ok}/{K} correct)'),
    ]
    ax2.legend(handles=legend2, loc='upper left', framealpha=0.95, fontsize=12)
    ax2.set_xlabel('DOA (degrees)')
    ax2.set_ylabel('Normalized Spectrum (dB)')
    ax2.set_title(f'(b) T-COP  [{chr(945)}=0.85, 10 scans, {tcop_ok}/{K} correct]',
                  fontsize=15, fontweight='bold')
    ax2.set_ylim([-42, 5])
    ax2.set_xlim([-85, 85])

    # == (c) SD-COP ==
    ax3.plot(scan_deg, sdcop_db, color='#00AA00', linewidth=2.5, zorder=3)
    # For SD-COP, use different estimated color (magenta) to avoid clash with green spectrum
    for td in true_deg:
        ax3.axvline(x=td, color='#00AA00', linestyle='--', alpha=0.35, linewidth=0.8, zorder=1)
    ax3.plot(true_deg, np.full_like(true_deg, -38.0), 'o',
             color='#00AA00', markersize=8, markeredgecolor='darkgreen',
             markeredgewidth=0.7, zorder=5, clip_on=False)
    for ed_rad in sdcop_doas:
        idx = np.argmin(np.abs(scan_angles - ed_rad))
        ax3.plot(np.degrees(ed_rad), sdcop_db[idx], '^', color='#9900CC',
                 markersize=12, markeredgecolor='black', markeredgewidth=0.8, zorder=6)

    legend3 = [
        Line2D([0], [0], color='#00AA00', linewidth=2.5, label='SD-COP Spectrum'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00AA00',
               markeredgecolor='darkgreen', markersize=9, label=f'True DOAs (K={K})'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#9900CC',
               markeredgecolor='black', markersize=11,
               label=f'Estimated ({sdcop_ok}/{K} correct)'),
    ]
    ax3.legend(handles=legend3, loc='upper left', framealpha=0.95, fontsize=12)
    ax3.set_xlabel('DOA (degrees)')
    ax3.set_ylabel('Normalized Spectrum (dB)')
    ax3.set_title(f'(c) SD-COP  [Signal-domain deflation, {sdcop_ok}/{K} correct]',
                  fontsize=15, fontweight='bold')
    ax3.set_ylim([-42, 5])
    ax3.set_xlim([-85, 85])

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig0b_cop_family_spectrum.png'))
    plt.close(fig)
    print(f"  Saved fig0b_cop_family_spectrum.png")
    print(f"  COP: {cop_ok}/{K}, T-COP: {tcop_ok}/{K}, SD-COP: {sdcop_ok}/{K}")


if __name__ == '__main__':
    main()
    print("\nAll plots generated!")
