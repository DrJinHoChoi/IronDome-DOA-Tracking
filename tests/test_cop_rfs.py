#!/usr/bin/env python3
"""Test COP-RFS: GM-PHD Filter with COP Spectrum Birth Intensity.

Tests:
1. Smoke test: basic PHD filter operation
2. Multi-target tracking with moving targets
3. COP-RFS vs conventional tracker comparison
4. T-COP + PHD feedback loop
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from iron_dome_sim.signal_model.array import UniformLinearArray
from iron_dome_sim.signal_model.signal_generator import generate_snapshots
from iron_dome_sim.doa import SubspaceCOP, TemporalCOP
from iron_dome_sim.tracking import (COPPHD, MultiTargetTracker,
                                     ConstantVelocity)
from iron_dome_sim.eval.metrics import rmse_doa, detection_rate


def test_smoke():
    """Basic smoke test: PHD filter processes scans without error."""
    print("=" * 60)
    print("Test 1: COP-RFS Smoke Test")
    print("=" * 60)

    M = 8
    K = 5
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 256
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)

    # Create COP-RFS
    cop = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    model = ConstantVelocity(dt=0.1, process_noise_std=0.01)
    phd = COPPHD(model, cop, survival_prob=0.95, detection_prob=0.90,
                 birth_weight=0.1, clutter_rate=2.0)

    true_doas = np.radians([-40, -20, 0, 20, 40])

    print(f"  Config: M={M}, K={K}, SNR={snr_db}dB, T={T}")
    print(f"  True DOAs: {np.degrees(true_doas)}")
    print()

    for scan_i in range(10):
        np.random.seed(42 + scan_i)
        X, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                     "non_stationary")
        estimates, doa_meas, spectrum = phd.process_scan(X, scan_angles)

        n_est = len(estimates)
        est_doas = [est[0][0] for est in estimates]
        total_w = phd.get_target_count()
        n_comp = len(phd.gm_components)

        print(f"  Scan {scan_i+1:2d}: targets={n_est}, components={n_comp}, "
              f"total_weight={total_w:.2f}, "
              f"DOA_meas={len(doa_meas)}")

    # Final evaluation
    final_doas = phd.get_doa_estimates()
    if len(final_doas) > 0:
        rmse_val, _ = rmse_doa(final_doas, true_doas)
        pd, _ = detection_rate(final_doas, true_doas)
        print(f"\n  Final: Pd={pd:.2f}, RMSE={np.degrees(rmse_val):.2f} deg, "
              f"Detected={len(final_doas)}/{K}")
    else:
        print(f"\n  Final: No targets extracted")

    print("  [OK]")
    print()


def test_moving_targets():
    """Test tracking of slowly moving targets."""
    print("=" * 60)
    print("Test 2: Moving Target Tracking")
    print("=" * 60)

    M = 8
    K = 4
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 256
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_scans = 20
    dt = 0.1

    # Create COP-RFS
    cop = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    model = ConstantVelocity(dt=dt, process_noise_std=0.01)
    phd = COPPHD(model, cop, survival_prob=0.95, detection_prob=0.90,
                 birth_weight=0.15, clutter_rate=1.0)

    # Moving targets: each moves at ~2 deg/scan
    base_doas = np.radians([-50, -20, 10, 40])
    rates = np.radians([2.0, -1.5, 1.0, -2.5])  # deg/scan

    print(f"  Config: {K} moving targets, {n_scans} scans, dt={dt}s")
    print(f"  Initial DOAs: {np.degrees(base_doas)}")
    print(f"  Angular rates: {np.degrees(rates)} deg/scan")
    print()

    track_history = []

    print(f"  {'Scan':>5s} | {'True DOAs':>40s} | {'Est':>4s} | {'Pd':>5s} | {'RMSE':>6s}")
    print("  " + "-" * 70)

    for scan_i in range(n_scans):
        true_doas = base_doas + rates * scan_i
        # Clip to FOV
        true_doas = np.clip(true_doas, -np.pi / 2 + 0.05, np.pi / 2 - 0.05)

        np.random.seed(42 + scan_i)
        X, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                     "non_stationary")
        estimates, doa_meas, _ = phd.process_scan(X, scan_angles)

        est_doas = phd.get_doa_estimates()

        if len(est_doas) > 0:
            rmse_val, _ = rmse_doa(est_doas, true_doas)
            pd, _ = detection_rate(est_doas, true_doas)
        else:
            rmse_val, pd = np.radians(90), 0.0

        true_str = ", ".join([f"{np.degrees(d):5.1f}" for d in true_doas])
        print(f"  {scan_i+1:5d} | {true_str:>40s} | {len(est_doas):4d} | "
              f"{pd:5.2f} | {np.degrees(rmse_val):6.2f}")

        track_history.append({
            'true': true_doas.copy(),
            'estimated': est_doas.copy() if len(est_doas) > 0 else np.array([]),
            'pd': pd,
            'rmse': np.degrees(rmse_val)
        })

    # Summary
    avg_pd = np.mean([h['pd'] for h in track_history[3:]])  # Skip first 3 (warmup)
    avg_rmse = np.mean([h['rmse'] for h in track_history[3:]])
    print(f"\n  Average (after warmup): Pd={avg_pd:.2f}, RMSE={avg_rmse:.2f} deg")
    print()


def test_tcop_phd_feedback():
    """Test T-COP + PHD feedback loop (full COP-RFS pipeline)."""
    print("=" * 60)
    print("Test 3: T-COP + PHD Feedback Loop")
    print("=" * 60)

    M = 8
    K = 6
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 5  # Low SNR to show feedback benefit
    T = 128
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_scans = 15

    true_doas = np.radians([-50, -30, -10, 10, 30, 50])

    # Without feedback: COP + standard PHD
    cop_no_fb = SubspaceCOP(array, rho=2, num_sources=K, spectrum_type="combined")
    model1 = ConstantVelocity(dt=0.1, process_noise_std=0.01)
    phd_no_fb = COPPHD(model1, cop_no_fb, survival_prob=0.95,
                        detection_prob=0.90, birth_weight=0.15)

    # With feedback: T-COP + PHD (automatic feedback)
    tcop_fb = TemporalCOP(array, rho=2, num_sources=K, alpha=0.85,
                           prior_weight=0.2)
    model2 = ConstantVelocity(dt=0.1, process_noise_std=0.01)
    phd_fb = COPPHD(model2, tcop_fb, survival_prob=0.95,
                     detection_prob=0.90, birth_weight=0.15)

    print(f"  Config: M={M}, K={K}, SNR={snr_db}dB (low), T={T}")
    print(f"  True DOAs: {np.degrees(true_doas)}")
    print()

    print(f"  {'Scan':>5s} | {'COP+PHD':>16s} | {'T-COP+PHD':>16s}")
    print(f"  {'':>5s} | {'Pd':>6s} {'RMSE':>8s} | {'Pd':>6s} {'RMSE':>8s}")
    print("  " + "-" * 45)

    for scan_i in range(n_scans):
        np.random.seed(42 + scan_i)
        X, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                     "non_stationary")

        # COP + PHD (no feedback)
        est1, _, _ = phd_no_fb.process_scan(X, scan_angles)
        doas1 = phd_no_fb.get_doa_estimates()
        pd1, _ = detection_rate(doas1, true_doas) if len(doas1) > 0 else (0, [])
        rmse1, _ = rmse_doa(doas1, true_doas) if len(doas1) > 0 else (np.radians(90), [])

        # T-COP + PHD (with feedback)
        np.random.seed(42 + scan_i)  # Same data
        X2, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                      "non_stationary")
        est2, _, _ = phd_fb.process_scan(X2, scan_angles)
        doas2 = phd_fb.get_doa_estimates()
        pd2, _ = detection_rate(doas2, true_doas) if len(doas2) > 0 else (0, [])
        rmse2, _ = rmse_doa(doas2, true_doas) if len(doas2) > 0 else (np.radians(90), [])

        print(f"  {scan_i+1:5d} | {pd1:6.2f} {np.degrees(rmse1):8.2f} | "
              f"{pd2:6.2f} {np.degrees(rmse2):8.2f}")

    print()


def test_target_birth_death():
    """Test PHD filter's ability to handle target birth and death."""
    print("=" * 60)
    print("Test 4: Target Birth & Death")
    print("=" * 60)

    M = 8
    array = UniformLinearArray(M=M, d=0.5)
    snr_db = 15
    T = 256
    scan_angles = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    n_scans = 20

    cop = SubspaceCOP(array, rho=2, spectrum_type="combined")
    model = ConstantVelocity(dt=0.1, process_noise_std=0.01)
    phd = COPPHD(model, cop, survival_prob=0.90, detection_prob=0.90,
                 birth_weight=0.2, clutter_rate=1.0,
                 prune_threshold=1e-4)

    print(f"  Scenario: targets appear and disappear over time")
    print()

    print(f"  {'Scan':>5s} | {'K_true':>6s} | {'True DOAs':>30s} | "
          f"{'K_est':>5s} | {'Pd':>5s} | {'RMSE':>6s}")
    print("  " + "-" * 75)

    for scan_i in range(n_scans):
        # Dynamic target configuration
        if scan_i < 5:
            K = 3
            true_doas = np.radians([-30, 0, 30])
        elif scan_i < 10:
            K = 5
            true_doas = np.radians([-40, -20, 0, 20, 40])
        elif scan_i < 15:
            K = 4
            true_doas = np.radians([-40, -20, 20, 40])  # center target dies
        else:
            K = 2
            true_doas = np.radians([-40, 40])  # only 2 remain

        cop.num_sources = K

        np.random.seed(42 + scan_i)
        X, _, _ = generate_snapshots(array, true_doas, snr_db, T,
                                     "non_stationary")
        estimates, doa_meas, _ = phd.process_scan(X, scan_angles)

        est_doas = phd.get_doa_estimates()
        if len(est_doas) > 0:
            rmse_val, _ = rmse_doa(est_doas, true_doas)
            pd, _ = detection_rate(est_doas, true_doas)
        else:
            rmse_val, pd = np.radians(90), 0.0

        true_str = ", ".join([f"{np.degrees(d):5.1f}" for d in true_doas])
        print(f"  {scan_i+1:5d} | {K:6d} | {true_str:>30s} | "
              f"{len(est_doas):5d} | {pd:5.2f} | {np.degrees(rmse_val):6.2f}")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("COP-RFS: GM-PHD Filter + COP Spectrum Birth Intensity")
    print("Novel Multi-Target Tracker for Underdetermined DOA")
    print("=" * 60)
    print()

    test_smoke()
    test_moving_targets()
    test_tcop_phd_feedback()
    test_target_birth_death()

    print("=" * 60)
    print("All COP-RFS tests completed.")
    print("=" * 60)
