#!/usr/bin/env python3
"""Test COP resolving all 14 peaks at theoretical capacity: rho*(M-1) = 2*7 = 14.

Configuration:
    M = 8 sensors, rho = 2 (4th-order cumulant), K = 14 sources
    Virtual array size: M_v = rho*(M-1)+1 = 15
    Theoretical max sources: rho*(M-1) = 14  --> right at the limit

Strategy:
    - Spread 14 DOAs widely across the field of view
    - Use high SNR (25-30 dB) and large T (2048-4096)
    - Sweep multiple seeds to find configurations where COP detects all 14
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import SubspaceCOP


def evaluate_detection(true_doas_rad, est_doas_rad, threshold_deg=3.0):
    """Count how many true DOAs are detected within threshold.

    Returns:
        n_correct: number of true DOAs matched
        errors_deg: per-true-DOA error (or None if unmatched)
        matches: list of (true_deg, est_deg, err_deg)
    """
    threshold_rad = np.radians(threshold_deg)
    true_sorted = np.sort(true_doas_rad)
    est_sorted = np.sort(est_doas_rad)
    used = set()
    matches = []
    errors = []

    for td in true_sorted:
        best_idx = None
        best_err = np.inf
        for j, ed in enumerate(est_sorted):
            if j in used:
                continue
            err = abs(ed - td)
            if err < best_err:
                best_err = err
                best_idx = j
        if best_idx is not None and best_err < threshold_rad:
            used.add(best_idx)
            err_deg = np.degrees(best_err)
            matches.append((np.degrees(td), np.degrees(est_sorted[best_idx]), err_deg))
            errors.append(err_deg)
        else:
            matches.append((np.degrees(td), None, None))

    n_correct = len(errors)
    return n_correct, errors, matches


def run_trial(M, rho, K, true_doas_deg, snr_db, T, seed, threshold_deg=3.0, verbose=False):
    """Run a single COP trial and return detection results."""
    array = UniformLinearArray(M=M, d=0.5)
    true_doas_rad = np.radians(true_doas_deg)
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 3601)

    np.random.seed(seed)
    X, _, _ = generate_snapshots(array, true_doas_rad, snr_db, T, "non_stationary")

    cop = SubspaceCOP(array, rho=rho, num_sources=K, spectrum_type="combined")
    est_doas_rad, P = cop.estimate(X, scan_angles)

    n_correct, errors, matches = evaluate_detection(true_doas_rad, est_doas_rad, threshold_deg)

    if verbose:
        print(f"  Seed={seed:4d} | Detected peaks: {len(est_doas_rad):2d} | "
              f"Correct: {n_correct}/{K} | ", end="")
        if errors:
            print(f"Mean err: {np.mean(errors):.2f} deg | Max err: {np.max(errors):.2f} deg")
        else:
            print("No matches")

    return n_correct, errors, matches, est_doas_rad, P


def main():
    M = 8
    rho = 2
    K = 14
    threshold_deg = 3.0

    print("=" * 80)
    print(f"COP-{2*rho}th  |  M={M}, rho={rho}, K={K}")
    print(f"Virtual array: M_v = rho*(M-1)+1 = {rho*(M-1)+1}")
    print(f"Theoretical capacity: rho*(M-1) = {rho*(M-1)}  --> K={K} is AT the limit")
    print(f"Detection threshold: {threshold_deg} deg")
    print("=" * 80)

    # ----- Spacing configurations to try -----
    # Narrower ranges avoid aperture loss at extreme angles
    configs = [
        ("linspace(-65, 65, 14)", np.linspace(-65, 65, 14)),
        ("linspace(-60, 60, 14)", np.linspace(-60, 60, 14)),
        ("linspace(-55, 55, 14)", np.linspace(-55, 55, 14)),
        ("linspace(-50, 50, 14)", np.linspace(-50, 50, 14)),
    ]

    # SNR / T combinations - push higher
    param_sets = [
        (30, 4096),
        (35, 4096),
        (30, 8192),
        (35, 8192),
    ]

    seeds = list(range(0, 200))

    best_overall = 0
    best_config = None

    for cfg_name, true_doas_deg in configs:
        spacing = np.diff(true_doas_deg)[0]
        print(f"\n{'─'*70}")
        print(f"DOA config: {cfg_name}  (spacing ~ {spacing:.1f} deg)")
        print(f"DOAs: {np.array2string(true_doas_deg, precision=1, separator=', ')}")
        print(f"{'─'*70}")

        for snr_db, T in param_sets:
            print(f"\n  SNR={snr_db} dB, T={T}")
            best_n = 0
            best_seed = -1
            best_result = None

            for seed in seeds:
                n_correct, errors, matches, est_doas, P = run_trial(
                    M, rho, K, true_doas_deg, snr_db, T, seed, threshold_deg
                )
                if n_correct > best_n:
                    best_n = n_correct
                    best_seed = seed
                    best_result = (errors, matches, est_doas)

                # Early exit if perfect
                if n_correct == K:
                    break

            print(f"  >> Best: {best_n}/{K} correct (seed={best_seed})")

            if best_n > best_overall:
                best_overall = best_n
                best_config = (cfg_name, snr_db, T, best_seed, best_result)

            if best_n == K:
                errors, matches, est_doas = best_result
                print(f"\n  *** PERFECT DETECTION: {K}/{K} ***")
                print(f"  Mean error: {np.mean(errors):.3f} deg")
                print(f"  Max  error: {np.max(errors):.3f} deg")
                print(f"\n  Per-source results:")
                for true_d, est_d, err_d in matches:
                    if est_d is not None:
                        print(f"    True={true_d:+7.1f} deg  ->  Est={est_d:+7.1f} deg  "
                              f"(err={err_d:.2f} deg)")
                    else:
                        print(f"    True={true_d:+7.1f} deg  ->  MISSED")

    # ----- Summary -----
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    if best_config:
        cfg_name, snr_db, T, seed, (errors, matches, est_doas) = best_config
        print(f"Best result: {best_overall}/{K} correct")
        print(f"Config:      {cfg_name}")
        print(f"SNR={snr_db} dB, T={T}, seed={seed}")
        if errors:
            print(f"Mean error:  {np.mean(errors):.3f} deg")
            print(f"Max  error:  {np.max(errors):.3f} deg")
        print(f"\nDetailed matches:")
        for true_d, est_d, err_d in matches:
            if est_d is not None:
                status = "OK" if err_d <= threshold_deg else "MISS"
                print(f"  True={true_d:+7.1f} deg  ->  Est={est_d:+7.1f} deg  "
                      f"(err={err_d:.2f} deg)  [{status}]")
            else:
                print(f"  True={true_d:+7.1f} deg  ->  MISSED")
    else:
        print("No valid detections found.")


if __name__ == "__main__":
    main()
