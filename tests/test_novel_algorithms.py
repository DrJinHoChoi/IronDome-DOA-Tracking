#!/usr/bin/env python3
"""Test T-COP and SD-COP novel algorithm extensions.

Benchmarks:
1. Basic functionality (smoke test)
2. K scaling (number of sources)
3. SNR robustness
4. Close-spacing resolution
5. Temporal accumulation benefit (T-COP)
6. Deflation capacity extension (SD-COP)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import (SubspaceCOP, TemporalCOP, SequentialDeflationCOP,
                                MUSIC, Capon)
from iron_dome_sim.eval.metrics import rmse_doa, detection_rate


def test_smoke():
    """Basic smoke test: all algorithms run without errors."""
    print("=" * 60)
    print("Test 1: Smoke Test")
    print("=" * 60)

    M = 8
    K = 6
    array = UniformLinearArray(M=M, d=0.5)
    true_doas = np.radians(np.linspace(-40, 40, K))
    snr_db = 15
    T = 256

    np.random.seed(42)
    X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)

    algorithms = {
        'COP-4th': SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined"),
        'T-COP-4th': TemporalCOP(array, rho=2, num_sources=K),
        'SD-COP-4th': SequentialDeflationCOP(array, rho=2, num_sources=K),
    }

    for name, alg in algorithms.items():
        try:
            doa_est, spectrum = alg.estimate(X, scan_angles)
            rmse_val, _ = rmse_doa(doa_est, true_doas)
            pd, _ = detection_rate(doa_est, true_doas)
            print(f"  {name:15s}: RMSE={np.degrees(rmse_val):6.2f} deg, "
                  f"Detected={len(doa_est)}/{K}, Pd={pd:.2f}  [OK]")
        except Exception as e:
            print(f"  {name:15s}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    print()


def test_k_scaling():
    """Test detection rate as K increases (key issue #1)."""
    print("=" * 60)
    print("Test 2: K Scaling (Source Count)")
    print("=" * 60)

    M = 8
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 256
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)

    K_values = [4, 6, 8, 10, 12, 14]

    print(f"  {'K':>3s} | {'COP Pd':>8s} {'COP RMSE':>10s} | "
          f"{'T-COP Pd':>8s} {'T-COP RMSE':>10s} | "
          f"{'SD-COP Pd':>8s} {'SD-COP RMSE':>10s}")
    print("  " + "-" * 80)

    for K in K_values:
        true_doas = np.radians(np.linspace(-55, 55, K))
        np.random.seed(42)
        X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")

        results = {}
        for alg_name, alg_cls, alg_kwargs in [
            ('COP', SubspaceCOP, dict(rho=2, num_sources=K, spectrum_type="combined")),
            ('T-COP', TemporalCOP, dict(rho=2, num_sources=K)),
            ('SD-COP', SequentialDeflationCOP, dict(rho=2, num_sources=K)),
        ]:
            alg = alg_cls(array, **alg_kwargs)
            try:
                doa_est, _ = alg.estimate(X, scan_angles)
                rmse_val, _ = rmse_doa(doa_est, true_doas)
                pd, _ = detection_rate(doa_est, true_doas)
                results[alg_name] = (pd, np.degrees(rmse_val))
            except Exception:
                results[alg_name] = (0.0, float('inf'))

        cop = results['COP']
        tcop = results['T-COP']
        sdcop = results['SD-COP']
        print(f"  {K:3d} | {cop[0]:8.2f} {cop[1]:10.2f} | "
              f"{tcop[0]:8.2f} {tcop[1]:10.2f} | "
              f"{sdcop[0]:8.2f} {sdcop[1]:10.2f}")

    print()


def test_snr_robustness():
    """Test performance across SNR range (key issue #3)."""
    print("=" * 60)
    print("Test 3: SNR Robustness")
    print("=" * 60)

    M = 8
    K = 8
    array = UniformLinearArray(M=M, d=0.5)
    true_doas = np.radians(np.linspace(-50, 50, K))
    T = 256
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)

    snr_values = [-5, 0, 5, 10, 15, 20]

    print(f"  {'SNR':>4s} | {'COP Pd':>8s} {'RMSE':>8s} | "
          f"{'T-COP Pd':>8s} {'RMSE':>8s} | "
          f"{'SD-COP Pd':>8s} {'RMSE':>8s}")
    print("  " + "-" * 72)

    for snr_db in snr_values:
        results = {}
        for alg_name, alg_cls, alg_kwargs in [
            ('COP', SubspaceCOP, dict(rho=2, num_sources=K, spectrum_type="combined")),
            ('T-COP', TemporalCOP, dict(rho=2, num_sources=K)),
            ('SD-COP', SequentialDeflationCOP, dict(rho=2, num_sources=K)),
        ]:
            alg = alg_cls(array, **alg_kwargs)
            # For T-COP, simulate temporal accumulation over 5 scans
            if alg_name == 'T-COP':
                for scan_i in range(5):
                    np.random.seed(42 + scan_i)
                    X_scan, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                                       "non_stationary")
                    doa_est, _ = alg.estimate(X_scan, scan_angles)
            else:
                np.random.seed(42)
                X, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                             "non_stationary")
                try:
                    doa_est, _ = alg.estimate(X, scan_angles)
                except Exception:
                    doa_est = np.array([])

            rmse_val, _ = rmse_doa(doa_est, true_doas)
            pd, _ = detection_rate(doa_est, true_doas)
            results[alg_name] = (pd, np.degrees(rmse_val))

        cop = results['COP']
        tcop = results['T-COP']
        sdcop = results['SD-COP']
        print(f"  {snr_db:4d} | {cop[0]:8.2f} {cop[1]:8.2f} | "
              f"{tcop[0]:8.2f} {tcop[1]:8.2f} | "
              f"{sdcop[0]:8.2f} {sdcop[1]:8.2f}")

    print()


def test_temporal_accumulation():
    """Test T-COP's temporal accumulation benefit specifically."""
    print("=" * 60)
    print("Test 4: Temporal Accumulation Benefit (T-COP specific)")
    print("=" * 60)

    M = 8
    K = 8
    array = UniformLinearArray(M=M, d=0.5)
    true_doas = np.radians(np.linspace(-50, 50, K))
    snr_db = 5  # Low SNR to show benefit
    T = 128  # Short snapshots
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)

    n_scans = 10
    tcop = TemporalCOP(array, rho=2, num_sources=K, alpha=0.85)

    print(f"  {'Scan':>5s} | {'Pd':>6s} {'RMSE':>8s} | Notes")
    print("  " + "-" * 45)

    for scan_i in range(n_scans):
        np.random.seed(42 + scan_i)
        X_scan, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                           "non_stationary")
        doa_est, _ = tcop.estimate(X_scan, scan_angles)
        rmse_val, _ = rmse_doa(doa_est, true_doas)
        pd, _ = detection_rate(doa_est, true_doas)

        notes = ""
        if scan_i == 0:
            notes = "(cold start)"
        elif scan_i == n_scans - 1:
            notes = "(fully accumulated)"

        print(f"  {scan_i+1:5d} | {pd:6.2f} {np.degrees(rmse_val):8.2f} | {notes}")

    # Compare with single-scan COP
    cop = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    np.random.seed(42)
    X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")
    doa_est, _ = cop.estimate(X, scan_angles)
    rmse_val, _ = rmse_doa(doa_est, true_doas)
    pd, _ = detection_rate(doa_est, true_doas)
    print(f"  {'COP':>5s} | {pd:6.2f} {np.degrees(rmse_val):8.2f} | (single scan baseline)")

    print()


def test_deflation_capacity():
    """Test SD-COP's extended capacity beyond single-stage COP."""
    print("=" * 60)
    print("Test 5: Deflation Capacity Extension (SD-COP specific)")
    print("=" * 60)

    M = 8
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 512
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)

    print(f"  COP max sources (rho=2): {array.max_sources(2)}")
    print(f"  Testing K values beyond single-stage capacity...\n")

    K_values = [7, 10, 14, 18, 21]

    print(f"  {'K':>3s} | {'COP Pd':>8s} {'RMSE':>8s} | "
          f"{'SD-COP Pd':>8s} {'RMSE':>8s} | {'SD stages':>10s}")
    print("  " + "-" * 65)

    for K in K_values:
        true_doas = np.radians(np.linspace(-60, 60, K))
        np.random.seed(42)
        X, _, _ = generate_snapshots(array, true_doas, snr_db, T, "non_stationary")

        # Standard COP
        cop = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
        try:
            doa_est_cop, _ = cop.estimate(X, scan_angles)
            rmse_cop, _ = rmse_doa(doa_est_cop, true_doas)
            pd_cop, _ = detection_rate(doa_est_cop, true_doas)
        except Exception:
            pd_cop, rmse_cop = 0.0, np.radians(90)

        # SD-COP
        sdcop = SequentialDeflationCOP(array, rho=2, num_sources=K,
                                        batch_size=5, max_stages=5)
        try:
            doa_est_sd, _ = sdcop.estimate(X, scan_angles)
            rmse_sd, _ = rmse_doa(doa_est_sd, true_doas)
            pd_sd, _ = detection_rate(doa_est_sd, true_doas)
            n_stages = len(sdcop.stage_results)
        except Exception as e:
            pd_sd, rmse_sd, n_stages = 0.0, np.radians(90), 0
            print(f"    SD-COP error for K={K}: {e}")

        print(f"  {K:3d} | {pd_cop:8.2f} {np.degrees(rmse_cop):8.2f} | "
              f"{pd_sd:8.2f} {np.degrees(rmse_sd):8.2f} | {n_stages:10d}")

    print()


def test_close_spacing():
    """Test close-spacing resolution (key issue #2)."""
    print("=" * 60)
    print("Test 6: Close-Spacing Resolution")
    print("=" * 60)

    M = 8
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 512
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)

    spacing_values = [10, 7, 5, 3, 2, 1]

    print(f"  {'Space':>6s} | {'COP Pd':>8s} {'RMSE':>8s} | "
          f"{'T-COP Pd':>8s} {'RMSE':>8s} | "
          f"{'SD-COP Pd':>8s} {'RMSE':>8s}")
    print("  " + "-" * 72)

    for spacing_deg in spacing_values:
        K = 3
        center = 0
        true_doas = np.radians([center - spacing_deg, center, center + spacing_deg])

        results = {}
        for alg_name, alg_cls, alg_kwargs in [
            ('COP', SubspaceCOP, dict(rho=2, num_sources=K, spectrum_type="combined")),
            ('T-COP', TemporalCOP, dict(rho=2, num_sources=K)),
            ('SD-COP', SequentialDeflationCOP, dict(rho=2, num_sources=K)),
        ]:
            alg = alg_cls(array, **alg_kwargs)
            if alg_name == 'T-COP':
                # Accumulate over 5 scans
                for scan_i in range(5):
                    np.random.seed(42 + scan_i)
                    X_scan, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                                       "non_stationary")
                    doa_est, _ = alg.estimate(X_scan, scan_angles)
            else:
                np.random.seed(42)
                X, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                             "non_stationary")
                doa_est, _ = alg.estimate(X, scan_angles)

            rmse_val, _ = rmse_doa(doa_est, true_doas)
            pd, _ = detection_rate(doa_est, true_doas)
            results[alg_name] = (pd, np.degrees(rmse_val))

        cop = results['COP']
        tcop = results['T-COP']
        sdcop = results['SD-COP']
        print(f"  {spacing_deg:5d}° | {cop[0]:8.2f} {cop[1]:8.2f} | "
              f"{tcop[0]:8.2f} {tcop[1]:8.2f} | "
              f"{sdcop[0]:8.2f} {sdcop[1]:8.2f}")

    print()


def test_tracker_feedback():
    """Test T-COP with simulated tracker feedback."""
    print("=" * 60)
    print("Test 7: T-COP with Tracker Feedback Loop")
    print("=" * 60)

    M = 8
    K = 8
    array = UniformLinearArray(M=M, d=0.5)
    true_doas = np.radians(np.linspace(-50, 50, K))
    snr_db = 5  # Low SNR to show feedback benefit
    T = 128
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)

    # T-COP without feedback
    tcop_no_fb = TemporalCOP(array, rho=2, num_sources=K, alpha=0.85,
                              prior_weight=0.0)
    # T-COP with feedback
    tcop_fb = TemporalCOP(array, rho=2, num_sources=K, alpha=0.85,
                           prior_weight=0.3)

    print(f"  {'Scan':>5s} | {'No Feedback':>16s} | {'With Feedback':>16s}")
    print(f"  {'':>5s} | {'Pd':>6s} {'RMSE':>8s} | {'Pd':>6s} {'RMSE':>8s}")
    print("  " + "-" * 45)

    prev_doas_fb = np.array([])

    for scan_i in range(8):
        np.random.seed(42 + scan_i)
        X_scan, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                           "non_stationary")

        # Set tracker feedback for the feedback version
        if len(prev_doas_fb) > 0:
            tcop_fb.set_tracker_predictions(prev_doas_fb,
                                             n_confirmed=len(prev_doas_fb))

        doa_no_fb, _ = tcop_no_fb.estimate(X_scan, scan_angles)
        doa_fb, _ = tcop_fb.estimate(X_scan, scan_angles)

        rmse_no, _ = rmse_doa(doa_no_fb, true_doas)
        pd_no, _ = detection_rate(doa_no_fb, true_doas)

        rmse_fb, _ = rmse_doa(doa_fb, true_doas)
        pd_fb, _ = detection_rate(doa_fb, true_doas)

        # Simulated tracker output = current DOA estimates (simplified)
        prev_doas_fb = doa_fb

        print(f"  {scan_i+1:5d} | {pd_no:6.2f} {np.degrees(rmse_no):8.2f} | "
              f"{pd_fb:6.2f} {np.degrees(rmse_fb):8.2f}")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Novel Algorithm Testing: T-COP & SD-COP")
    print("IP Extensions of 2rho-th order Subspace COP")
    print("=" * 60)
    print()

    test_smoke()
    test_k_scaling()
    test_snr_robustness()
    test_temporal_accumulation()
    test_deflation_capacity()
    test_close_spacing()
    test_tracker_feedback()

    print("=" * 60)
    print("All tests completed.")
    print("=" * 60)
