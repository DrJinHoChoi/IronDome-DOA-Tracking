#!/usr/bin/env python3
"""Compare COP vs MUSIC across realistic non-stationary signal types.

Key point: COP exploits non-Gaussianity via higher-order cumulants.
Different signal types have different degrees of non-Gaussianity,
which directly affects COP's advantage over conventional methods.

Signal types tested:
  - stationary: Complex Gaussian i.i.d. (baseline - COP should NOT help)
  - non_stationary: AM sinusoid
  - speech: Quasi-periodic voiced segments (SmartEAR application)
  - fm: Frequency modulation
  - chirp: Linear FM (radar)
  - psk: QPSK communication
  - missile: Doppler + Swerling I RCS
  - mixed: Random per-source
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import SubspaceCOP, MUSIC


def count_correct(est, true, thr_deg=3.0):
    if len(est) == 0:
        return 0
    return sum(1 for d in true if min(abs(est - d)) < np.radians(thr_deg))


def rmse(est, true):
    """RMSE between matched estimated and true DOAs."""
    if len(est) == 0:
        return 90.0
    from scipy.optimize import linear_sum_assignment
    cost = np.abs(est[:, None] - true[None, :])
    r, c = linear_sum_assignment(cost)
    errors = cost[r, c]
    return np.degrees(np.sqrt(np.mean(errors ** 2)))


def main():
    M = 8
    array = UniformLinearArray(M=M, d=0.5)
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 3601)

    signal_types = [
        "stationary",       # Gaussian baseline
        "non_stationary",   # AM sinusoid
        "speech",           # Speech-like
        "fm",               # Frequency modulation
        "chirp",            # LFM radar
        "psk",              # QPSK communication
        "missile",          # Doppler + Swerling I
        "mixed",            # Random mix
    ]

    # ===== Experiment 1: Underdetermined (K=10 > M-1=7) =====
    print("=" * 100)
    print("Experiment 1: COP vs MUSIC across signal types")
    print(f"  M={M}, K=10 (underdetermined), SNR=15 dB, T=512, 5 seeds")
    print("=" * 100)

    K = 10
    snr_db = 15
    T = 512
    seeds = [42, 55, 77, 123, 300]
    true_doas = np.radians(np.linspace(-54, 54, K))

    print(f"\n{'Signal Type':>16}  {'COP det':>8}  {'MUSIC det':>10}  "
          f"{'COP RMSE':>9}  {'MUSIC RMSE':>11}  {'COP advantage':>14}")
    print("-" * 100)

    for sig_type in signal_types:
        cop_dets = []
        music_dets = []
        cop_rmses = []
        music_rmses = []

        for seed in seeds:
            np.random.seed(seed)
            X, _, _ = generate_snapshots(array, true_doas, snr_db, T, sig_type)

            # COP-4th (rho=2)
            cop = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
            cop_est, _ = cop.estimate(X, scan_angles)
            cop_dets.append(count_correct(cop_est, true_doas))
            cop_rmses.append(rmse(cop_est, true_doas))

            # MUSIC (limited to M-1=7)
            music = MUSIC(array, num_sources=min(K, M - 1))
            music_est, _ = music.estimate(X, scan_angles)
            music_dets.append(count_correct(music_est, true_doas))
            music_rmses.append(rmse(music_est, true_doas))

        avg_cop = np.mean(cop_dets)
        avg_music = np.mean(music_dets)
        avg_cop_r = np.mean(cop_rmses)
        avg_music_r = np.mean(music_rmses)
        advantage = avg_cop - avg_music

        print(f"{sig_type:>16}  {avg_cop:>5.1f}/{K}  {avg_music:>7.1f}/{K}  "
              f"{avg_cop_r:>8.2f}d  {avg_music_r:>10.2f}d  "
              f"{'+'if advantage>0 else ''}{advantage:>5.1f} sources")

    # ===== Experiment 2: Determined case (K=6 < M-1=7) =====
    print("\n\n" + "=" * 100)
    print("Experiment 2: Determined case (K=6) - COP should match MUSIC")
    print(f"  M={M}, K=6, SNR=15 dB, T=512, 5 seeds")
    print("=" * 100)

    K2 = 6
    true_doas2 = np.radians(np.linspace(-50, 50, K2))

    print(f"\n{'Signal Type':>16}  {'COP det':>8}  {'MUSIC det':>10}  "
          f"{'COP RMSE':>9}  {'MUSIC RMSE':>11}")
    print("-" * 80)

    for sig_type in signal_types:
        cop_dets = []
        music_dets = []
        cop_rmses = []
        music_rmses = []

        for seed in seeds:
            np.random.seed(seed)
            X, _, _ = generate_snapshots(array, true_doas2, snr_db, T, sig_type)

            cop = SubspaceCOP(array, rho=2, num_sources=K2, spectrum_type="combined")
            cop_est, _ = cop.estimate(X, scan_angles)
            cop_dets.append(count_correct(cop_est, true_doas2))
            cop_rmses.append(rmse(cop_est, true_doas2))

            music = MUSIC(array, num_sources=K2)
            music_est, _ = music.estimate(X, scan_angles)
            music_dets.append(count_correct(music_est, true_doas2))
            music_rmses.append(rmse(music_est, true_doas2))

        print(f"{sig_type:>16}  {np.mean(cop_dets):>5.1f}/{K2}  "
              f"{np.mean(music_dets):>7.1f}/{K2}  "
              f"{np.mean(cop_rmses):>8.2f}d  {np.mean(music_rmses):>10.2f}d")

    # ===== Experiment 3: Gaussian signal (COP should NOT help) =====
    print("\n\n" + "=" * 100)
    print("Experiment 3: Gaussian vs Non-Gaussian signal comparison")
    print("  Key: COP advantage comes ONLY from non-Gaussianity")
    print("  Gaussian signals have zero higher-order cumulants => COP = noise")
    print("=" * 100)

    K3 = 10
    true_doas3 = np.radians(np.linspace(-54, 54, K3))

    for sig_type in ["stationary", "speech", "missile", "mixed"]:
        cop_dets = []
        for seed in seeds:
            np.random.seed(seed)
            X, _, _ = generate_snapshots(array, true_doas3, snr_db, T, sig_type)
            cop = SubspaceCOP(array, rho=2, num_sources=K3, spectrum_type="combined")
            cop_est, _ = cop.estimate(X, scan_angles)
            cop_dets.append(count_correct(cop_est, true_doas3))
        print(f"  {sig_type:>16}: COP detects {np.mean(cop_dets):.1f}/{K3} "
              f"(seeds: {cop_dets})")

    print("\n" + "=" * 100)
    print("KEY INSIGHT: COP advantage is strongest for non-Gaussian signals")
    print("  - Gaussian (stationary): COP cumulant is NOISE-ONLY => fails")
    print("  - Speech, FM, chirp, PSK, missile: strong non-Gaussianity")
    print("    => COP cumulant contains SOURCE structure => resolves K > M-1")
    print("=" * 100)


if __name__ == "__main__":
    main()
